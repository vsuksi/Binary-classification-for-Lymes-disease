def test(**kwargs):
    '''
    Test 006: does calcom.utils.generate_partitions() give
        expected partitions (up to some reasonable tolerance)
        for a few test cases?
    '''

    import calcom
    import numpy as np

    # Reproducibility
    np.random.seed(314159)

    # Will be populated with true/false for each subtest.
    subtests = []

    ########################
    #
    # Subtest 0: balanced classes; 5-fold cross-validation.
    #
    subtests.append(True)
    n = 100
    nfolds = 5
    tol = 0.05  # class representation should match entire dataset to within this tolerance

    labels = np.repeat([['lion','tiger']], n)  # 100 lion, then 100 tiger

    partitions = calcom.utils.generate_partitions(labels, method='stratified_k-fold', nfolds=nfolds)
    for i,part in enumerate(partitions):
        train_set, test_set = part

        # Do the training and validation folds represent the overal ratio of the classes, approximately?
        l_count = list(labels[train_set]).count('lion')
        tr_ratio = float(l_count)/len(train_set)

        l_count = list(labels[test_set]).count('lion')
        te_ratio = float(l_count)/len(test_set)

        if (np.abs(tr_ratio - 0.5)<tol) and (np.abs(te_ratio - 0.5)<tol):
            continue
        else:
            # failure
            subtests[-1] = False
        #
    #

    ########################
    #
    # Subtest 1: imbalanced subclasses; 5-fold cross-validation.
    #
    subtests.append(True)
    nfolds = 5
    tol = 0.05  # class representation should match entire dataset to within this tolerance

#    labels = np.repeat([['lion','tiger']], n)  # 100 lion, then 100 tiger
    labels = np.concatenate( ( np.repeat('lion',160), np.repeat('tiger',40) ) )
    true_l_ratio = float( list(labels).count('lion') ) / len(labels)

    partitions = calcom.utils.generate_partitions(labels, method='stratified_k-fold', nfolds=nfolds)
    for i,part in enumerate(partitions):
        train_set, test_set = part
        # Do the training and validation folds represent the overal ratio of the classes, approximately?
        l_count = list(labels[train_set]).count('lion')
        tr_ratio = float(l_count)/len(train_set)

        l_count = list(labels[test_set]).count('lion')
        te_ratio = float(l_count)/len(test_set)

        if (np.abs(tr_ratio - true_l_ratio)<tol) and (np.abs(te_ratio - true_l_ratio)<tol):
            continue
        else:
            # failure
            subtests[-1] = False
        #
    #

    ########################
    #
    # Subtest 2: imbalanced subclasses; no clean ratios. Relatively large dataset.
    #
    n = 347 # happens to be prime
    ratio_target = 0.83123

    labels = np.array([{True:'lion',False:'tiger'}[np.random.rand()<ratio_target] for _ in range(n) ])
    true_l_ratio = float( list(labels).count('lion') ) / len(labels)

    partitions = calcom.utils.generate_partitions(labels, method='stratified_k-fold', nfolds=nfolds)
    for i,part in enumerate(partitions):
        train_set, test_set = part
        # Do the training and validation folds represent the overal ratio of the classes, approximately?
        l_count = list(labels[train_set]).count('lion')
        tr_ratio = float(l_count)/len(train_set)

        l_count = list(labels[test_set]).count('lion')
        te_ratio = float(l_count)/len(test_set)

        if (np.abs(tr_ratio - true_l_ratio)<tol) and (np.abs(te_ratio - true_l_ratio)<tol):
            continue
        else:
            # failure
            subtests[-1] = False
        #
    #

    # TODO - a few more cases that could be checked...

    return all(subtests)
#

if __name__=="__main__":
    print( test() )
