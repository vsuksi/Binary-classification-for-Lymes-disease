import numpy as np
from copy import copy
from calcom.classifiers._centroidencoder import deepANNClassifier as dc
from calcom.classifiers._centroidencoder.utilityDBN import standardizeData
from calcom.classifiers._abstractclassifier import AbstractClassifier

class NeuralnetworkClassifier(AbstractClassifier):
    def __init__(self):
        '''
        Setup default parameters
        '''
        self.params = {}
        self.testClassifier = None

        self.params['auto_layer_structure'] = False

        self.params['hLayer'] = [25]
        self.params['actFunc'] = ['tanh']
        self.params['errorFunc'] = 'CE'
        self.params['optimizationFuncName'] = 'scg'
        self.params['noItrPre'] = 100
        self.params['noItrPost'] = 40
        self.params['noItrSoftmax'] = 100
        self.params['noItrFinetune'] = 100
        self.params['batchFlag'] = False
        # self.params['standardizeData'] = True

        # Output data. Might be overwritten.
        self.results = {}
        self.results['pred_labels'] = []

    @property
    def _is_native_multiclass(self):
        return True
    #
    @property
    def _is_ensemble_method(self):
        return False
    #

    def initParams(self,hLayer,actFunc,errorFunc,optimizationFuncName,noItrPre,noItrPost,batchFlag):
        self.params = {}
        self.params['hLayer'] = hLayer
        self.params['actFunc'] = actFunc
        self.params['errorFunc'] = errorFunc
        self.params['optimizationFuncName'] = optimizationFuncName
        self.params['noItrPre'] = noItrPre
        self.params['noItrPost'] = noItrPost
        self.params['batchFlag'] = batchFlag
        self.params['auto_layer_structure'] = False

    def feval(self,fName,*args):
        return eval(fName)(*args)

    def tanh(self,X):
        return np.tanh(X)

    def sigmoid(self,X):
        '''
        Numerically-stable sigmoid function.

                Y=copy(X)
                Y[Y>=0]=1.0/(1+np.exp(-Y[Y>=0]))
                Y[Y<0]=np.exp(Y[Y<0])/(1+np.exp(Y[Y<0]))
                return Y

                .. note:
                    Taken from http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        '''
        return expit(X) #using the sigmoidal of scipy package

    def softplus(self,X):
        return np.log(1+np.exp(X))

    def rect(self,X):
        X[X<=0]=0*X[X<=0]
        return X

    def rectl(self,X):
        X[X<=0]=0.01*X[X<=0]
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
        ##pdb.set_trace()
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

    def decision_function(self,data):
        lOut=[standardizeData(data,self._mu,self._std)]

        lLength=len(self.testClassifier.netW)

        for j in range(lLength):
            d = np.dot(lOut[-1],self.testClassifier.netW[j][1:,:])+self.testClassifier.netW[j][0]#first row in the weight is the bias
            #Take the activation function from the dictionary and apply it
            lOut.append(self.feval('self.'+self.testClassifier.actFunc[j],d) if j<lLength-1 else d)
        fOut = self.calcLogProb(lOut[-1])
        return fOut

    def pseudoLoss(self,trData,trLabels):
        labels = (trLabels == np.unique(trLabels)).astype(int)
        flipLabels = -(labels-1) # This will convert '1's into '0's and '0's into '1's
        probs = self.decision_function(trData)
        pLoss = np.sum(flipLabels*probs,axis=1)
        return pLoss

    def _predict(self,data):
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
        ##pdb.set_trace()
        predictedTstLabel=(np.argmax(fOut,axis=1)).reshape(-1,1)

        predictedTstLabel = predictedTstLabel.flatten()
        # self.results['pred_labels'] = predictedTstLabel.flatten()
        # return predictedTstLabel,fOut

        # pred_labels = self._process_output_labels(predictedTstLabel)
        #
        # return pred_labels
        return predictedTstLabel
    #

    def _fit(self,trData,trLabels):
        # Generate automatic deep layer structure if indicated.
        n,d = np.shape(trData)
        # internal_labels = self._process_input_labels(trLabels)

        if self.params['auto_layer_structure']:
            d1 = min( int(np.sqrt(d)), d)
            d2 = min( max(3,int(np.sqrt(d1))), d1)  # blah
            self.params['hLayer'] = [d1,d2]
            self.params['actFunc'] = ['tanh','tanh']
        #

        self._mu,self._std,trData = standardizeData(trData)

        dict1 = {}
        dict1['inputL'] = np.shape(trData)[1]
        # dict1['outputL'] = len(np.unique(internal_labels))
        dict1['outputL'] = len(np.unique(trLabels))
        dict1['hL'] = self.params['hLayer']
        dict1['actFunc'] = copy(self.params['actFunc'])
        dict1['nItr'] = [self.params['noItrPre'] for i in range(len(self.params['hLayer']))]
        dict1['errorFunc'] = self.params['errorFunc']
        nnc = dc.deepClassifier(dict1)
        ##pdb.set_trace()

        # temp = np.array(internal_labels)
        temp = np.array(trLabels)
        temp.shape = (len(temp),1)  # Convert to column vector for deepANN code

        optimizationFuncNamePre,optimizationFuncNamePost = self.params['optimizationFuncName'],self.params['optimizationFuncName']
        windowSize,batchSize,noEpoch,cvThreshold,batchFlag = 0,0,0,0,self.params['batchFlag']
        nnc.preTrain(trData,temp,[],trData,temp,optimizationFuncNamePre,cvThreshold,windowSize,self.params['noItrPre'],freezeLayerFlag=True)

        ##pdb.set_trace()
        # print('Post training classifier')
        # print('Network configuration:',dict1['inputL'],'-->',dict1['hL'],'-->',dict1['outputL'])
        # def postTrain(self,trData,trLabels,indVar=[],valData=[],valLabels=[],tstData,tstLabels,optimizationFuncName='scg',windowSize=0,nItr=10,batchFlag=False,batchSize=100,noEpoch=500):

        nnc.postTrain(trData,temp,[],trData,temp,trData,temp,optimizationFuncNamePost,windowSize,self.params['noItrPost'],batchFlag,batchSize,noEpoch)

        predictedTrLabel,noErrorTr = nnc.test(trData,temp)

        self.testClassifier = dc.deepClassifier(dict1)
        self.testClassifier.preTrainingDone = True
        self.testClassifier.netW = copy(nnc.netW)
        #return
        return self
    #

    def visualize(self,*args):
        pass

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
