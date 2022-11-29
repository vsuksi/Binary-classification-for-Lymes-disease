def test(**kwargs):
    '''
    Test 013: How does CCExperiment handle complex cross-validation
        schemes?

        What about when a class isn't represented in the
        test set's validation and the training set allows for
        both sets in the predicted labels?
    '''
    import calcom
    import numpy as np

    seed = kwargs.get('seed', 2718281828)
    nfolds = kwargs.get('nfolds', 5)
    cutoff = kwargs.get('cutoff', 0.7)

    ccd = calcom.utils.synthetic_datasets.generate_synthetic_ccd2(seed=seed)

    subtests = []

    #
    try:
        rc = calcom.classifiers.RandomClassifier()
        ssvm = calcom.classifiers.SSVMClassifier()
        classifiers = [rc, ssvm]

        metric = calcom.metrics.ConfusionMatrix('bsr')

        # training and testing set are expected to have
        # proportions from the two classes reflective of the
        # entire dataset. This functionality has already
        # been tested in a prior test of calcom.utils.generate_partitions.
        cross_validation = 'leave-one-attrvalue-out'

        cce = calcom.CCExperiment(
            classifiers,
            ccd,
            metric,
            classification_attr='friendly',
            cross_validation=cross_validation,
            cross_validation_attr='subject_id',
            verbosity=0,
            save_all_classifiers=True,
            folds=nfolds
        )

        result = cce.run()

        subtests.append( True )
    except:
        subtests.append( False )
    #

    return all(subtests)
#

if __name__=="__main__":
    print( test() )
