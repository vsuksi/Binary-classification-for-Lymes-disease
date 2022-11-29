def test(**kwargs):
    '''
    Test 010: Can we perform a basic cross-validation experiments
        on the synthetic dataset using calcom.Experiment?
        Are these experiments doing the expected thing for a
        few choices of simple parameters?
    '''
    import calcom
    import numpy as np

    subtests = []

    seed = kwargs.get('seed', 2718281828)
    nfolds = kwargs.get('nfolds', 5)
    cutoff = kwargs.get('cutoff', 0.7)

    np.random.seed(seed=seed)

    try:
        # expect to pass this if the previous test passed.
        ccd = calcom.utils.synthetic_datasets.generate_synthetic_ccd1(seed=seed)
        subtests.append( True )
    except:
        subtests.append( False )
    #

    try:
        data = ccd.generate_data_matrix()
        labels = ccd.generate_labels('animal')

        rc = calcom.classifiers.RandomClassifier()
        ssvm = calcom.classifiers.SSVMClassifier()
        classifiers = [rc, ssvm]

        metric = calcom.metrics.ConfusionMatrix('bsr')

        # training and testing set are expected to have
        # proportions from the two classes reflective of the
        # entire dataset. This functionality has already
        # been tested in a prior test of calcom.utils.generate_partitions.
        cross_validation = 'stratified_k-fold'

        cce = calcom.Experiment(
            data,
            labels,
            classifiers,
            cross_validation,
            metric,
            folds=nfolds,
            verbosity=0,
            save_all_classifiers=True
        )

        result = cce.run()

        subtests.append( True )
    except:
        subtests.append( False )
    #

    # study the results for the SSVMClassifier.
    try:
        ssvm_models = result['SSVMClassifier']
        ssvm_results = cce.classifier_results['SSVMClassifier_1']
        subtests.append( True )
    except:
        subtests.append( False )
    #

    # Are there the expected number of folds?
    try:
        subtests.append( len(ssvm_models)==nfolds )
    except:
        subtests.append( False )
    #
    try:
        subtests.append(
            len(ssvm_results['scores'])==nfolds
        )
    except:
        subtests.append( False )
    #

    # Are the prediction rates above 70% (say)? Note that
    # this dataset (as of April 15, 2019) is almost completely
    # separable. SSVM should do well enough in separation.
    # It will succeed in this for the default seed, at the very least.
    try:
        subtests.append( all( ssvm_results['scores'] >= cutoff ) )
    except:
        subtests.append( False )
    #

    return all( subtests )
#

if __name__=="__main__":
    print( test() )
