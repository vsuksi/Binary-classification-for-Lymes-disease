
def createVoronoiRegion(data,**kwargs):
    return(KMean(data,**kwargs))
#

def KMean(data,**kwargs):
    '''
    K-means implementation hooking into sklearn.cluster.KMeans.

    Inputs:
        data - m-by-n data matrix
    Outputs:
        vRegion - ??? (TODO: FILL IN)

    Optional inputs:
        nCenter - integer, number of centers to use for kmeans. (default: len(data)/3)
        nIter - integer, max number of iterations to use for sklearn.cluster.KMeans() (default: 500)
        verbosity - integer, controlling amount of output. Value of 0 means no output. (defaut: 0)
    '''
    nCenter = kwargs.get('nCenter', max(1,len(data)//3) )
    nItr = kwargs.get('nItr',500)
    verbosity = kwargs.get('verbosity',0)

    from sklearn.cluster import KMeans

    if verbosity>0: print('Running kMeans clustering with no of centers:',nCenter)

    kmeans = KMeans(n_clusters=nCenter, n_init=50, max_iter=nItr, random_state=0).fit(data)
    centers = kmeans.cluster_centers_

    vRegion = {}
    for c in range(nCenter):
        key = 'Region'+str(c+1)
        vRegion[key] = {}
        vRegion[key]['Center'] = kmeans.cluster_centers_[c]
        vRegion[key]['PatternIndex'] = np.where(c == kmeans.labels_)[0]
    #
    vRegion['Inertia']=np.sqrt(kmeans.inertia_)/nCenter #Inertia=Sum of the squared distances of samples from their nearest cluster center.

    if verbosity>1: print('Inertia per cluster:',vRegion['Inertia'])

    return vRegion
#

def createCEData(trData,trLabels,**kwargs):
    '''
    Inputs:
        trData - m-by-n data matrix
        trLabels - list-like size m of corresponding labels
    Outputs:
        trInput,trOutput - ??? (TODO: FILL IN)

    Optional inputs:
        nCenter - integer, number of centers to use for kmeans. (default: len(data)/3)
        verbosity - integer, controlling amount of output. Value of 0 means no output. (defaut: 0)

        Other optional inputs get passed to createVoronoiRegion and KMean.
    '''
    nCenter = kwargs.get('nCenter', max(1,len(data)//3) )
    verbosity = kwargs.get('verbosity',0)

    import numpy as np

    if nCenter==1:
        #Now create input and output for centroidencoder. Output is a representative for each class. I'm taking the centroid.
        trInput=[]
        trOutput=[]
        valInput=[]
        valOutput=[]
        trClassVal=[]
        valClassVal=[]
        for c in np.unique(trLabels):
            tmpD = trData[np.where(trLabels==c)[0],:]

            noP=np.shape(tmpD)[0]
            trOutput.append(np.tile(np.mean(tmpD,axis=0),(noP,1)))
            trInput.append(tmpD)
        #
        trInput=np.vstack((trInput))
        trOutput=np.vstack((trOutput))
    else:
        trInput=[]
        trOutput=[]
        vRegion=createVoronoiRegion(trData,**kwargs)
        for k in vRegion.keys():
            if 'Region' in k:
                d=trData[vRegion[k]['PatternIndex'],:]
                noP=np.shape(d)[0]
                trInput.append(d)
                trOutput.append(np.tile(vRegion[k]['Center'],(noP,1)))
        trInput=np.vstack((trInput))
        trOutput=np.vstack((trOutput))
    return trInput,trOutput
#

def generateData(trData,trLabels,**kwargs):
    '''
    Inputs:
        trData - m-by-n data matrix
        trLabels - list-like size m of corresponding labels
    Outputs:
        syntheticData[:,:-1],syntheticData[:,-1].reshape(-1,1)
        syntheticData - ??? (TODO: FILL IN)
        ??? - ??? (TODO: FILL IN)

    Optional inputs:
        All optional inputs get passed to createCEData, KMean, etc.
        nCERun - integer; number of centroidencoder runs to do (??? TODO: EXPLAIN)

        nCenter - integer; number of centers to use for K-means algorithm (default: max(1,len(trData)//3))
        nItr - integer; max number of iterations to use for K-means algorithm (default: 500)
        verbosity - integer; level of output

        params - A dictionary controlling the structure of the neural networks used; see
            calcom/calcom/classifiers/centroidencoder/Autencoder.py for the options.
            Any values not in params.keys() are set to be defaults from
            calcom.classifiers.CentroidencoderClassifier().
    '''
    from calcom.classifiers.centroidencoder import Autoencoder as ae
    from calcom.classifiers.centroidencoder import scaledconjugategradient as scg
    import copy

    m,n = np.shape(trData)

    nCERun = kwargs.get('nCERun', 1)
    nCenter = kwargs.get('nCenter', max(1,m//3) )
    nItr = kwargs.get('nItr',500)
    verbosity =  kwargs.get('verbosity',0)
    params = kwargs.get('params',{})

    # ugh
    keys = list(params.keys())
    defaultce = calcom.classifiers.CentroidencoderClassifier()
    defaultce_params = dict(defaultce.params)

    for cekey in list(defaultce_params.keys()):
        if cekey not in keys:
            params[cekey] = defaultce_params[cekey]
    #

    syntheticData=[]

    trInput,trOutput=createCEData(trData,trLabels,**kwargs)

    ########################################## Centroidencoder layerwise pre-training ##########################################
    dict1={}
    dict1['inputL'] = n
    dict1['outputL'] = n
    dict1['hL'] = params['hLayer']
    dict1['actFunc'] = params['actFunc']
    dict1['nItr']= params['noItrPre']*np.ones(len(dict1['actFunc'])).astype(int)
    dict1['errorFunc'] = params['errorFunc']

    cvThreshold,windowSize = 0,0;
    bottleneckPre=ae.BottleneckSAE(dict1)
    if verbosity>0:
        print('Network configuration:',dict1['inputL'],'-->',dict1['hL'],'-->',dict1['outputL'])
        print('Layer-wise pre-training the bottle-neck neural network.')
    #def train(iData,oData,valInput,valOutput,optimizationFuncName,cvThreshold,windowSize,nItr=10,weightPrecision=0,errorPrecision=0,verbose=False,freezeLayerFlag=True):
    bottleneckPre.train(trInput,trOutput,trInput,trOutput,params['optimizationFuncName'],cvThreshold,windowSize,params['noItrPre'])

    ##########################################            Centroidencoder post-training         ##########################################
    # for r in range(nCERun):
    dict_post={}
    dict_post['inputL'] = n
    dict_post['outputL'] = n
    dict_post['hL'] = params['hLayer']
    dict_post['actFunc'] = copy.copy(params['actFunc'])
    dict_post['actFunc'].extend(['linear'])
    dict_post['nItr'] = params['noItrPost']
    dict_post['errorFunc'] = params['errorFunc']
    optimizationFuncNamePost = params['optimizationFuncName']
    bottleneckPost = ae.BottleneckAE(dict_post)
    windowSize = 0
    batchFlag = params['batchFlag']

    bottleneckPost.netW = copy.copy(bottleneckPre.netW)


    if verbosity>1:
        print('Post training bottle-neck neural network',str(dict_post['inputL']),str(dict_post['hL']),str(dict_post['outputL']))
    #def train(iData,oData,valInput,valOutput,optimizationFuncName,windowSize=0,nItr=10,dropPercentage=0,weightPrecision=0,errorPrecision=0,verbose=False):
    bottleneckPost.train(trInput,trOutput,trInput,trOutput,optimizationFuncNamePost,windowSize,params['noItrPost'])

    #Take the output of centroid-encoder and stack it with original trainind data to enlarge the trsining set.
    ceOutput = bottleneckPost.regenDWOStandardize(trData)[-1]
    # syntheticData.append(np.hstack((ceOutput,trLabels)))
    syntheticData = ceOutput

    #With the added new ce output recalculate centroids.
    trInput,trOutput=createCEData(trData,trLabels)

    #Save weight from pre-training
    params['preW'] = copy.copy(bottleneckPost.netW)

    ############################################### Centroidencoder post-training ###############################################
    del bottleneckPre,bottleneckPost,params['preW']


    # return syntheticData[:,:-1],syntheticData[:,-1].reshape(-1,1)
    return syntheticData
#

if __name__ == "__main__":
    from matplotlib import pyplot
    import numpy as np
    import calcom

    m = 20
    p = 1273 # Not needed right now
    np.random.seed(57721)   # Euler-Mascheroni

    params = {'hLayer':[2,2,2],
                'noItrPre':15,
                'noItrPost':150,
                'noItrSoftmax':25,
                'noItrFinetune':50,
                'batchFlag':'False'}

    props = dict(boxstyle='square', facecolor=[0.95,0.95,0.95], edgecolor='k', alpha=0.5)

    th = 2*np.pi*np.random.rand(m)
    x = np.cos(th)
    y = np.sin(th)

    data = np.vstack((x,y)).T

    data_synth_all = []
    for j,nCenters in enumerate([3,4,8,m]):
        data_synth = np.empty((0,2))
        for i in range(p//m+1):
            ds = generateData(data,[0 for i in range(m)], nCenter=nCenters, params=params)
            data_synth = np.vstack((data_synth,ds))
        #
        data_synth_all.append(data_synth)
        print(j,nCenters)
    #

    fig,ax = pyplot.subplots(1,4, sharex=True, sharey=True, figsize=(14,4))
    for j,nCenters in enumerate([3,4,8,m]):
        ax[j].scatter(data_synth_all[j][:,0], data_synth_all[j][:,1], c='r', s=1)
        ax[j].scatter(data[:,0],data[:,1], c='k', marker=r'$\odot$', s=100)

        ax[j].set_title('nCenters=%i'%nCenters)

        for col,lab,marker,si in [['k','Original data',r'$\odot$',100],['r','Synthetic data','.',20]]:
            ax[j].scatter([],[],c=col,label=lab, marker=marker, s=si)
        #
        ax[j].axis('equal')
    #

    ax[3].legend(loc='upper right', fontsize=10)

    fig.tight_layout()

    fig.suptitle('CE synthetic data', fontsize=18)
    fig.subplots_adjust(top=0.85)

    pyplot.show(block=False)
#
