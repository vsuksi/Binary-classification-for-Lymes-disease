def test(**kwargs):
    '''
    Test 016: Can we instantiate and run IFR with the 
    default parameters on a small, randomly generated test dataset?
    '''
    import calcom
    import numpy as np 

    # fix the random seed
    np.random.seed(2718281828)

    data = np.random.rand(10,10)
    labels = np.random.choice([0,1], 10)

    try:
        ifr = calcom.preprocessors.IterativeFeatureRemoval.IFR()
        ifr.params['verbosity'] = 0
        features = ifr.process(data,labels)
        return True
    except:
        print('IFR instantiation and ifr.process fail.')
        return False
     #
#
