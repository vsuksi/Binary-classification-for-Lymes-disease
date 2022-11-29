from __future__ import absolute_import, division, print_function
from calcom.visualizers._abstractvisualizer import AbstractVisualizer


class CentroidencoderVisualizer(AbstractVisualizer):

    def __init__(self):
        '''
        Setup default parameters (a dictionary). Defaults are:
        params['dim'] = 2
        '''
        self.params = {}
        self.params['dim'] = 2     # Number of dimensions (2 or 3 only)
    #

    def project(self,data,labels, **kwargs):
        '''
        Centroid encoder visualizer. Note that this is a *supervised*
        visualization technique; that is, label information is required
        of the data, as a neural net needs to be trained to map the data
        to their class's centroid on a training set, then validated
        on a testing set.

        Inputs:
            data: numpy array, n-by-m, where n is the number of observations
                and m is the dimensionality of the data.
            labels: array of size n, where n is the number of observations
        Optional inputs:
            readable_label_map: Mapping dictionary. From numbers to meaningful strings.
            dim: dimensionality of projection; override self.params['dim']

        Outputs:
            coords: numpy array n-by-d, where d is the number of dimensions
                of the projection.
        '''
        import numpy as np
        from calcom.classifiers._centroidencoder.utilityDBN import standardizeData

        self.params['dim'] = kwargs.get('dim', self.params['dim'])

        # TODO: REMOVE HARD-CODED PARAMETERS HERE
        splitRatio = 0.8
        labeledData = []
        labeledData.append(np.hstack((data,labels.reshape(-1,1))))
        labeledData = np.vstack((labeledData))
        train_set,test_set = splitData(labeledData,splitRatio)
        train,trainLabels,test,testLabels = train_set[:,:-1],train_set[:,-1].reshape(-1,1),test_set[:,:-1],test_set[:,-1].reshape(-1,1)

        outputCentroid = calcCentroid(train,trainLabels)

        mu1,std1,standardizeInput = standardizeData(train)#training(input) data standardization
        mu2,std2,standardizeOutput = standardizeData(outputCentroid)#training(output) data standardization
        standardizeTest = standardizeData(test,mu1,std1)#test data standardization

        # TODO: REMOVE HARD-CODED PARAMETERS HERE.
        #Centroidencoder starts
        # hLayer = [25,3,25]
        hLayer = [25, self.params['dim'] ,25]
        actFuncList = ['tanh','linear','tanh']
        iterations,scgItr=1,40
        plotLabel = 'Centroidencoder Visualizer'
        errorTrace,projectedTrData,projectedTstData = Centroidencoder(standardizeInput,standardizeOutput,standardizeTest,scgItr,hLayer,actFuncList)
        self.trainLabels = trainLabels
        self.testLabels = testLabels
        # self.readable_label_map = readable_label_map
        self.readable_label_map = kwargs.get('label_map', {})

        # return (projectedTrData,projectedTstData)
        return np.vstack( (projectedTrData,projectedTstData) )
    #

    def visualize(self,coords):
        '''
        Input: n by d array of (projected) data.
        '''
        import calcom.plot_wrapper as plotter

        FutureWarning('This visualization will be modified in the near future.')

        d = self.params['dim']
        if (d==2 or d==3):
            plotter.scatterTrainTest(
                coords[0],
                self.trainLabels,
                coords[1],
                self.testLabels,
                readable_label_map=self.readable_label_map,
                title="Centroidencoder Visualizer",
                dim=d
                )
        else:
            print('Error: only d=2 and d=3 are supported for scatterplotting PCA')
            return None,None
        #
    #

#

def splitData(labeled_data,split_ratio):
    '''
    This method will split the dataset into training and test set based on the split ratio.
    Training and test set will have data from each class according to the split ratio.
    First hold the data of different classes in different variables.

    TODO: can we just use calcom.utils.generate_partitions?
    '''
    import numpy as np

    train_set =[]
    test_set =[]
    no_data = len(labeled_data)
    sorted_data = labeled_data[np.argsort(labeled_data[:,-1])]#sorting based on the numeric label.
    first_time = 'Y'
    for classes in np.unique(sorted_data[:,-1]):
        temp_class = np.array([sorted_data[i] for i in range(no_data) if sorted_data[i,-1] == classes])
        np.random.shuffle(temp_class)#Shuffle the data so that we'll get variation in each run
        tr_samples = np.floor(len(temp_class)*split_ratio)
        tst_samples = len(temp_class) - tr_samples
        if(first_time == 'Y'):
            train_set = temp_class[:int(tr_samples),]
            test_set = temp_class[-int(tst_samples):,]
            first_time = 'N'
        else:
            train_set = np.vstack((train_set,temp_class[:int(tr_samples),]))
            test_set = np.vstack((test_set,temp_class[-int(tst_samples):,]))
    return train_set,test_set
#

def calcCentroid(data,labels):
    import numpy as np

    trClassVal = []
    trOutput = []
    for c in np.unique(labels):
        tmpD = data[np.where(labels==c)[0],:]
        trClassVal.append(np.mean(tmpD,axis=0))
        noP = np.shape(tmpD)[0]
        trOutput.append(np.tile(np.mean(tmpD,axis=0),(noP,1)))
    return np.vstack((trOutput))

def Centroidencoder(inputData,outputData,tstData,scgItr,hLayer,actFuncList):
    '''
    Single-purpose centroidencoder.
    TODO: Can we merge this with components of the centroid encoder
        in calcom.classifiers? Would prefer not to deal with managing multiple
        implementation.
    '''
    from . import Autoencoder as ae
    import numpy as np

    dataDim = np.shape(inputData)[1]
    dict={}
    dict['inputL']=dataDim
    dict['outputL']=dataDim
    dict['hL']=hLayer
    dict['actFunc']=actFuncList
    dict['nItr']=[5,5,5]
    dict['errorFunc']='MSE'
    centroidEncoding=ae.Autoencoder(dict)

    optimizationFuncName = 'scg'
    windowSize=0
    postItr=0
    postBatchItr=scgItr
    batchFlag=False
    batchSize=0
    noEpoch=0
    errThreshold=0.04
    print('Network configuration:',dict['inputL'],'-->',dict['hL'],'-->',dict['outputL'])
    print('Post training starts.')
    #def postTrain(self,trData,trLabels,indVar=[],valData=[],valLabels=[],optimizationFuncName='scg',nItr=10,batchFlag=False,batchSize=100,noEpoch=500)

    centroidEncoding.postTrain(inputData,outputData,inputData,outputData,optimizationFuncName,windowSize,postBatchItr,batchFlag,batchSize,noEpoch,errThreshold)
    l=np.ceil(len(hLayer)/2).astype(int)
    projectedTrData = centroidEncoding.regenDWOStandardize(inputData)[l]
    projectedTstData = centroidEncoding.regenDWOStandardize(tstData)[l]
    return centroidEncoding.postTrErr,projectedTrData,projectedTstData
#
