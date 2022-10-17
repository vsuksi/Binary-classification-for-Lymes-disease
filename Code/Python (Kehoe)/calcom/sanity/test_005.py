def test(**kwargs):
    '''
    Test 005: does calcom.utils.generate_partitions() generate
        partitions with no intersection between training and testing
        portions?

        Also, are classes well-represented in the training/testing
        set for various scenarios (e.g. balanced or imbalanced
        classes, or one-attrvalue-out)?
    '''

    import calcom
    import numpy as np

    # Reproducibility
    np.random.seed(314159)

    # Will be populated with true/false for each subtest.
    subtests = []

    ##############
    #
    # Subtest 0: balanced classes, large n, stratified_k-fold
    #
    n = 100
    labels = np.repeat([['lion','tiger']],n)
    partitions = calcom.utils.generate_partitions(labels,method='stratified_k-fold',nfolds=5)

    check0 = check_partition_intersection(partitions)

    # check that there's representation of the minority class
    # in every training/testing fold; this must happen with stratified k-fold.
    check1 = check_label_representation(partitions,labels)

    subtests.append( check0 and check1 )

    ##############
    #
    # Subtest 1: imbalanced classes, large n, stratified_k-fold
    #
    labels = np.concatenate( ( np.repeat('lion',160), np.repeat('tiger',40) ) )
    partitions = calcom.utils.generate_partitions(labels,method='stratified_k-fold',nfolds=5)

    check0 = check_partition_intersection(partitions)
    check1 = check_label_representation(partitions,labels)

    subtests.append( check0 and check1 )

    ##############
    #
    # Subtest 2: imbalanced classes, stratified_k-fold, with minority class having just
    #   enough representation to cover each fold, possibly with a little more.
    #
    labels = np.concatenate( ( np.repeat('lion',177), np.repeat('tiger',23) ) )
    partitions = calcom.utils.generate_partitions(labels,method='stratified_k-fold',nfolds=10)

    check0 = check_partition_intersection(partitions)
    check1 = check_label_representation(partitions,labels)

    subtests.append( check0 and check1 )

    ##############
    #
    # Subtest 3: plain k-fold, four folds, roughly balanced classes.
    #
    labels = np.concatenate( ( np.repeat('lion',117), np.repeat('tiger',83) ) )
    partitions = calcom.utils.generate_partitions(labels,method='k-fold',nfolds=4)

    subtests.append( check_partition_intersection(partitions) )

    ##############
    #
    # Subtest 4: plain k-fold, four folds, highly imbalanced classes.
    #
    labels = np.concatenate( ( np.repeat('lion',187), np.repeat('tiger',13) ) )
    partitions = calcom.utils.generate_partitions(labels,method='k-fold',nfolds=10)

    subtests.append( check_partition_intersection(partitions) )

    ##############
    #
    # Subtest 5: leave-one-out
    #
    labels = np.random.choice(['lion','tiger'],200)
    partitions = calcom.utils.generate_partitions(labels,method='leave-one-out')

    subtests.append( check_partition_intersection(partitions) )

    ##############
    #
    # Subtest 6: stratified_k-fold, leave-one-attr-value-out
    #
    labels = np.concatenate( ( np.repeat('lion',187), np.repeat('tiger',13) ) )
    attrvalues = np.repeat([['subj%i'%i for i in range(40)]] , 5)
    partitions = calcom.utils.generate_partitions(labels,method='leave-one-attrvalue-out',attrvalues=attrvalues)

    check0 = check_partition_intersection(partitions)
    check1 = all( [len( np.intersect1d(attrvalues[train], attrvalues[test]) )==0 for (train,test) in partitions] )

    subtests.append( check0 and check1 )

    ###########################################

    return all(subtests)
#

def check_partition_intersection(partitions_in):
    '''
    helper function, verifies no intersections
    '''
    import numpy as np

    check = True
    for (train,test) in partitions_in:
        intersection = np.intersect1d(train,test) # intersection between training and testing data
        check = (check and len(intersection)==0)
    #
    return check
#

def check_label_representation(partitions_in, labels_in):
    '''
    helper function, verifies representation (relevant for stratified k-fold)
    '''
    import numpy as np

    classes = np.unique(labels_in)
    check = True
    for part in partitions_in:
        for j,thing in enumerate(part):
            representation = [cl in labels_in[thing] for cl in classes]
            check = check and all(representation)
    #
    return check
#

if __name__=="__main__":
    print( test() )
