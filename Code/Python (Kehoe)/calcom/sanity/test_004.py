def test(**kwargs):
    '''
    Test 004: Do all the classifiers have the required components?
        Every classifier should have implemented the functions:
            ._fit()
            ._predict()
            ._is_native_multiclass()
    '''

    import calcom
    required = ['_fit','_predict','_is_native_multiclass', '_is_ensemble_method']

    # reference class
    abstract_clf = calcom.classifiers._abstractclassifier.AbstractClassifier

    subs = calcom.classifiers.__dict__

    properly_implemented = {}


    for s in subs:
        # Only care about non-underscore
        if str(s[0])!='_':
            try:
                # Only care about classes
                instantiation = subs[s]()
            except:
                continue
            #

            # If we're at this point - check for the essentials.
            all_implemented = all( [hasattr(instantiation, r) for r in required] )

            properly_implemented[s] = all_implemented
    #

    return all(properly_implemented.values())
#

if __name__=="__main__":
    print( test() )
