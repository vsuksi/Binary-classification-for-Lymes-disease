def test(**kwargs):
    '''
    Test 007: Can every classifier (other than ensemble
        methods) actually fit/predict (nevermind the
        accuracy)?
    '''
    verbosity = kwargs.get('verbosity',0)

    import calcom
    import numpy as np

    # Reproducibility
    np.random.seed(314159)

    data_class_a = np.repeat(np.array([[1.,0.]]), 50, axis=0)
    data_class_b = np.repeat(np.array([[-1.,0.]]), 50, axis=0)

    data = np.vstack( [data_class_a, data_class_b] )
    data[:,1] = np.random.randn( data.shape[0] )
    labels = np.concatenate( [np.repeat('a',50), np.repeat('b',50)] )

    # shuffle the data
    shuffle = np.random.permutation(100)
    data = data[shuffle]
    labels = labels[shuffle]

    n,d = np.shape(data)

    # Will be populated with true/false for each subtest.
    subtests = []

    subs = calcom.classifiers.__dict__

    if verbosity>0: print('')

    for s in subs.keys():
        # Only care about non-underscore
        #if verbosity>0: print('\t'+str(s)+' : ', end='')
#        import pdb
#        pdb.set_trace()
        if str(s[0])=='_':
            continue
        else:
            if verbosity>0: print('\t'+str(s)+' : ', end='')
            try:
                classifier = subs[s]()
            except:
                if verbosity>0: print('failed to instantiate.')
                subtests.append( False )
                continue
            #
            if classifier._is_ensemble_method:
                # skip
                if verbosity>0: print('ensemble method; skipped.')
                continue
            #
            try:
                # fit/predict with a single 2-fold.
                classifier.fit( data[:n//2], labels[:n//2] )
                pred = classifier.predict( data[n//2:] )

                # are the predictions within labels seen in the training?
                if len( np.setdiff1d( np.unique(pred), np.unique(labels[:n//2]) ) )!=0:
                    if verbosity>0: print('predicted labels are outside those in the training set.')
                    subtests.append( False )
                    continue
                else:
                    if verbosity>0: print('passed')
                    subtests.append( True )
                    continue
                #
            except:
                # something went wrong
                if verbosity>0: print('failed to fit, predict, or something else.')
                subtests.append( False )
                continue
            #
        #
    #

    return all(subtests)
#

if __name__=="__main__":
    print( test() )
