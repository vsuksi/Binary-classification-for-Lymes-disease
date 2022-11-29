'''
Author: Tomojit Ghosh
Script Name: sparsecentroidencoderclassifier.py
Details: This script implements the sparsified version of centroid-encoder classifier. The sparsity is applied to 
         extract features from input data. A sparsity promoting layet (SPL) is added between the input layer
         and the first hidden layer. Each node in SPL has a one-to-one weighted connection with the corresponding
         node in the input layer. Nodes in the SPL layer don't have any bias. No of nodes in the input layer is same 
         as the number of nodes in the input layer. L1 penalty is applied on the weights connecting the input layer
         and the SPL layer. This penalty will drive lot of weights to zero and the corresponding input node can be ignored.
         Input nodes with higher connection weight can be thought of important features.
'''
#from __future__ import absolute_import, division, print_function
from calcom.classifiers._abstractclassifier import AbstractClassifier
import numpy as np
import pdb

class SparseCentroidencoderClassifier(AbstractClassifier):
    def __init__(self):
        '''
        Setup default parameters
        '''
        self.params = {}
        self.testClassifier = None

        # If true, the 'hLayer' and 'actFunc' are OVERWRITTEN when self.fit()
        # is called to a heuristic sqrt scaling with three hidden layers.
        # For example, if input layer is size 10**4, then the hLayer structure
        # is [10**2, 10**1, 10**2].
        self.params['auto_layer_structure'] = False
        self.params['bottleneckArch'] = False
        self.params['hLayer'] = [100,100]
        self.params['actFunc'] = ['rect','rect']
        self.params['errorFunc'] = 'MSE'
        self.params['l1Penalty'] = 0.001
        self.params['optimizationFuncName'] = 'scg'
        self.params['noItrPre'] = 10
        self.params['noItrPost'] = 10
        self.params['noItrSoftmax'] = 10
        self.params['noItrFinetune'] = 40
        self.params['batchFlag'] = False
        self.params['standardizeData'] = True
        #variables for normalized gradient amplification
        self.params['sampleWeight']=[]
        self.params['weightedErrorFlag']=False
        self.params['sumError']=None

        # Output data. Might be overwritten.
        self.results = {}
        self.results['pred_labels'] = []

        super().__init__()
    #

    @property
    def _is_native_multiclass(self):
        return True
    #
    @property
    def _is_ensemble_method(self):
        return False

    def initParams(self,hLayer,actFunc,l1Penalty,errorFunc,optimizationFuncName,noItrPre,noItrPost,noItrSoftmax,noItrFinetune,batchFlag,bottleneckFlag):
        #self.params = {}
        self.params['hLayer'] = hLayer
        self.params['actFunc'] = actFunc
        self.params['l1Penalty'] = l1Penalty
        self.params['errorFunc'] = errorFunc
        self.params['optimizationFuncName'] = optimizationFuncName
        self.params['noItrPre'] = noItrPre
        self.params['noItrPost'] = noItrPost
        self.params['noItrSoftmax'] = noItrSoftmax
        self.params['noItrFinetune'] = noItrFinetune
        self.params['batchFlag'] = batchFlag
        self.params['bottleneckArch'] = bottleneckFlag

    def _fit(self,trData,trLabels):
        '''
        trData: Training Data
        trLabels: Training Labels
        '''

        from calcom.classifiers._centroidencoder import Autoencoder as ae
        from calcom.classifiers._centroidencoder import deepANNClassifier as dc
        from calcom.classifiers._centroidencoder import FeatureSelectingFramework as fsf
        from calcom.classifiers._centroidencoder.utilityDBN import standardizeData
        from copy import copy
        
#        internal_labels = self._process_input_labels(trLabels)
        internal_labels = trLabels

        n,d = np.shape(trData)
        if self.params['auto_layer_structure']:
            d1 = min( int(np.sqrt(d)), d)
            d2 = min( max(3,int(np.sqrt(d1))), d1)  # blah
            moo = [d1,d2,d1]
            self.params['hLayer'] = [d1,d2,d1]
            self.params['actFunc'] = ['tanh','tanh','tanh']
        #


        # trLabels = trLabels.reshape(-1,1)
        internal_labels = np.array(internal_labels).reshape(-1,1)
        #pdb.set_trace()
        self._mu,self._std,trData = standardizeData(trData)

        #Update params
        self.params['actFunc'].insert(0,'SPL')
        self.params['hLayer'].insert(0,trData.shape[1])

        #Now create input and output for centroidencoder. Output is a representative for each class. I'm taking the centroid.
        trInput=[]
        trOutput=[]
        valInput=[]
        valOutput=[]
        trClassVal=[]
        valClassVal=[]
        # for c in np.unique(trLabels):
        for c in self._label_info['unique_labels_mapped']:
            tmpD = trData[np.where(internal_labels==c)[0],:]
            #print('No of samples in class ',c,' is ',len(tmpD))
            noP=np.shape(tmpD)[0]
            trOutput.append(np.tile(np.mean(tmpD,axis=0),(noP,1)))
            trInput.append(tmpD)

        trInput=np.vstack((trInput))
        trOutput=np.vstack((trOutput))
        #################################### Sparse CE layerwise pre-training and post-training #####################################
        dict1={}
        dict1['inputL'] = np.shape(trData)[1]
        dict1['outputL'] = np.shape(trData)[1]
        dict1['hL'] = self.params['hLayer']
        dict1['actFunc'] = self.params['actFunc']
        dict1['l1Penalty'] = self.params['l1Penalty']
        dict1['outputActivation'] = 'linear'
        dict1['nItr']= self.params['noItrPre']*np.ones(len(dict1['actFunc'])).astype(int)
        dict1['errorFunc'] = self.params['errorFunc']
        dict1['nItrPre'] = self.params['noItrPre']
        dict1['nItrPost'] = self.params['noItrPost']
        
        cvThreshold,windowSize = 0,0;
        if self.params['bottleneckArch']:
            print('TBD')
        else:
            sparseCE=ae.SparseAutoencoder(dict1)
        if self.params['weightedErrorFlag']:
            sparseCE.weightedErrorFlag=True

        sparseCE.train(trInput,trOutput,trInput,trOutput,self.params['optimizationFuncName'])
        #################################### Sparse CE layerwise pre-training and post-training #####################################


        ###############################################   Linear Classifier Softmax   ###############################################
        newTrData=sparseCE.regenDWOStandardize(trData)[-2]

        #noTstData = np.shape(tstData)[0]
        optimizationFuncName = self.params['optimizationFuncName']
        windowSize = 0
        cvThreshold = 0
        noEpoch = 0
        batchFlag = self.params['batchFlag']
        batchSize = 0
        postBatchItr = self.params['noItrSoftmax']
        errThreshold = 0.025
        dict_softmax = {}
        dict_softmax['inputL'] = np.shape(newTrData)[1]
        dict_softmax['outputL'] = len(self._label_info['unique_labels_mapped'])
        dict_softmax['hL'] = []
        dict_softmax['actFunc'] = []
        dict_softmax['nItr'] = []
        dict_softmax['errorFunc'] = 'CE'
        softmax = dc.deepClassifier(dict_softmax)
        # softmax.postTrain(newTrData,trLabels,[],newTrData,trLabels,newTrData,trLabels,optimizationFuncName,windowSize,postBatchItr,batchFlag,batchSize,noEpoch,errThreshold)
        softmax.postTrain(newTrData,internal_labels,[],newTrData,internal_labels,newTrData,internal_labels,optimizationFuncName,windowSize,postBatchItr,batchFlag,batchSize,noEpoch,errThreshold)
        #Save weight from softmax layer
        self.params['softmaxW'] = copy(softmax.netW)
        # predictedTrLabel,noErrorTr=softmax.test(newTrData,trLabels)
        predictedTrLabel,noErrorTr=softmax.test(newTrData,internal_labels)
        #print('No. of misclassification on training data after softmax=',noErrorTr)
        ############################################### Linear Classifier Softmax ###############################################

        ###############################################         Fine tuning       ###############################################
        #Now backpropagate error from softmax layer thru hidden layers.
        # print('Starting fine tuning.')
        
        dict2 = {}
        dict2['inputL'] = np.shape(trData)[1]
        # dict2['outputL'] = len(np.unique(trLabels))
        dict2['outputL'] = len(self._label_info['unique_labels_mapped'])
        if self.params['bottleneckArch']:
            dict2['hL'] = dict_post['hL'][:int(np.ceil((len(dict_post['hL'])/2)))]
            dict2['actFunc'] = dict_post['actFunc'][:int(np.ceil((len(dict_post['actFunc'])/2)))]
        else:
            dict2['hL'] = dict1['hL'][1:]
            dict2['actFunc'] = dict1['actFunc'][1:]
        dict2['nItr'] = []
        dict2['errorFunc'] = 'CE'
        finetune=fsf.FeatureSelectingFramework(dict2)
        finetune.preTrainingDone = True
        #pdb.set_trace()
        tmpW=[]
        if self.params['bottleneckArch']:
            for l in range(int(len(dict_post['actFunc'])/2)):
                tmpW=copy(bottleneckPost.netW[l])
        else:
            tmpW=copy(sparseCE.netW[:-1])
        tmpW.append(copy(softmax.netW[0]))
        #print('Fine tuning...')
        finetune.setPreTrainedWeight(tmpW)
        noEpoch=0
        windowSize=50
        batchFlag=False
        postBatchItr=self.params['noItrFinetune']
        postItr=0
        errThreshold=1.0e-12
        optimizationFuncName='scg'
        if self.params['weightedErrorFlag']:
            finetune.weightedErrorFlag=True
        # finetune.postTrain(trData,trLabels,[],trData,trLabels,trData,trLabels,optimizationFuncName,windowSize,postBatchItr,batchFlag,batchSize,noEpoch,errThreshold)
        finetune.postTrainWithPretrainedWs(trData,internal_labels,trData,internal_labels,[],[],optimizationFuncName,windowSize,postBatchItr,batchFlag,batchSize,noEpoch,errThreshold)
        
        #Save weight from fine layer
        self.params['finetuneW'] = copy(finetune.netW)

        ############################################### Fine tuning ###############################################

        self.testClassifier=fsf.FeatureSelectingFramework(dict2)
        self.testClassifier.preTrainingDone=True
        self.testClassifier.netW = copy(finetune.netW)
        self.testClassifier.splW = copy(finetune.splW)
        #pdb.set_trace()
        return self

    def _predict(self,data):
        #data: test data
        '''
        lOut=[standardizeData(data,self._mu,self._std)]
        lLength=len(self.testClassifier.netW)
        for j in range(lLength):
            d = np.dot(lOut[-1],self.testClassifier.netW[j][1:,:])+self.testClassifier.netW[j][0]#first row in the weight is the bias
            #Take the activation function from the dictionary and apply it
            lOut.append(self.feval('self.'+self.testClassifier.actFunc[j],d) if j<lLength-1 else d)
        fOut = self.calcLogProb(lOut[-1])
        '''
        fOut = self.decision_function(data)
        predictedTstLabel=(np.argmax(fOut,axis=1)).reshape(-1,1)

        predictedTstLabel = predictedTstLabel.flatten()
        # self.results['pred_labels'] = predictedTstLabel

        # pred_labels = self._process_output_labels(predictedTstLabel)
        #
        # return pred_labels
        return predictedTstLabel
    #

    def decision_function(self,data):
        from calcom.classifiers._centroidencoder.utilityDBN import standardizeData

        lOut=[standardizeData(data,self._mu,self._std)]
        tmpTrData = lOut[-1]*np.tile(self.testClassifier.splW,(np.shape(data)[0],1))
        lOut.append(tmpTrData)
        lLength=len(self.testClassifier.netW)
        for j in range(lLength):
            d = np.dot(lOut[-1],self.testClassifier.netW[j][1:,:])+self.testClassifier.netW[j][0]#first row in the weight is the bias
            #Take the activation function from the dictionary and apply it
            lOut.append(self.feval('self.'+self.testClassifier.actFunc[j],d) if j<lLength-1 else d)
        fOut = self.calcLogProb(lOut[-1])
        return fOut

    def feval(self,fName,*args):
        return eval(fName)(*args)

    def tanh(self,X):
        return np.tanh(X)

    def sigmoid(self,X):
        '''
        #"Numerically-stable sigmoid function."
        #Taken from http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        Y=copy(X)
        Y[Y>=0]=1.0/(1+np.exp(-Y[Y>=0]))
        Y[Y<0]=np.exp(Y[Y<0])/(1+np.exp(Y[Y<0]))
        return Y
        '''
        return expit(X) #using the sigmoidal of scipy package

    def softplus(self,X):
        return np.log(1+np.exp(X))

    def rect(self,X):
        X[X<0]=0*X[X<0]
        return X

    def rectl(self,X):
        X[X<0]=0.01*X[X<0]
        return X

    def linear(self,X):
        return X

    def calcLogProb(self,cOut):
        '''
        target=np.exp(cOut)
        denom=np.repeat(np.sum(target,1).reshape(-1,1),self.outputDim,axis=1)
        return target/denom
        '''

        #Trick to avoid Overflow using Chuck's code
        #pdb.set_trace()
        mx = np.max(cOut)
        cOut=np.exp(cOut-mx)
        denom=np.exp(-mx) + np.sum(cOut,axis=1).reshape((-1,1))
        rowsHavingZeroDenom = denom == 0.0
        if np.sum(rowsHavingZeroDenom) > 0:
            Yshape = (cOut.shape[0],cOut.shape[1])
            nClasses = Yshape[1]
            Y = np.ones(Yshape) * 1.0/nClasses + np.random.uniform(0,0.1,Yshape)
            Y /= np.sum(Y,1).reshape((-1,1))
        else:
            Y=cOut/denom
        return Y


    def visualize(self,*args):
        pass

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
