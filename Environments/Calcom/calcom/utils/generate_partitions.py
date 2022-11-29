def generate_partitions(labels,method='stratified_k-fold',nfolds=5, attrvalues=[]):
    '''
    Returns tuple (or a list of tuples) of pointer arrays to data indicating the
    training and test data. Depending on the type of cross-validation desired,
    extra parameters can be provided.

    This assumes a subset of the original labels have already been reserved
    for comparison between classifiers, so only training and testing
    partitions need to be made.

    Currently implemented:

        method='k-fold'
        ----------------
            Used for k-fold cross validation. The original data is
            partitioned into k disjoint subsets of (roughly) equal size.
            The union of (k-1) subsets are used for training data,
            and 1 subset is used to test.

        method='stratified_k-fold'
        ---------------------
            Used for stratified k-fold cross validation. Similar to k-fold, but
            the subsets have approximately equal proportions of samples from each
            label as in the overall set, if possible.

        method='leave_one_out'
        ---------------------
            Essentially a k-fold cross validation, where the number of folds is
            the same as the number of data points. That is, with r data
            points, (r-1) data points are used to train the model(s), and
            testing is done on the last data point; repeating over all data.

    Examples:
    ---------------

    import calcom
    import numpy

    labels = numpy.array([0,0,0,1,1,1,2,2,2,2,2,2])

    partitions=calcom.utils.generate_partitions(labels,method='k-fold',nfolds=2)

        Example output (shuffling is done so results will vary):

        [[array([ 9,  3, 11,  4,  1,  2]), array([ 8,  6,  5, 10,  7,  0])],
        [array([ 8,  6,  5, 10,  7,  0]), array([ 9,  3, 11,  4,  1,  2])]]

    partitions=calcom.utils.generate_partitions(labels,method='stratified_k-fold',nfolds=3)

        Example output:

        [[array([ 3,  2, 10,  6,  4,  8, 11,  1]), array([9, 7, 5, 0])],
         [array([ 9,  7,  5,  0,  4,  8, 11,  1]), array([ 3,  2, 10,  6])],
         [array([ 9,  7,  5,  0,  3,  2, 10,  6]), array([ 4,  8, 11,  1])]]

    '''
    import numpy as np

    partitions = []
    n = len(labels)

    # Construct the set of subsets to be used.
    if (method=='k-fold'):
        if nfolds < 2:
            raise ValueError("Number of folds have to be greater than 1")
        # Create a shuffling of the indices then
        # just take consecutive blocks.
        shuffling = np.random.permutation(n)

        subsets = []
        # Cute trick to loop over folds of unequal sizes.
        idxs = np.linspace(0,n,nfolds+1).astype(int)
        for i in range(nfolds):
            il,ir = idxs[i],idxs[i+1]
            subsets.append(shuffling[il:ir])
            # subsets[i].sort()
        #
    elif (method=='stratified_k-fold'):
        if nfolds < 2:
            raise ValueError("Number of folds have to be greater than 1")
        # Shuffle the labels themselves, then build the
        # index set from these.
        perm = np.random.permutation(labels)

        # Need to be more careful about the permutation.
        # Build a list of lists. Each of these lists points to all
        # locations in perm with the same label.
        equivclasses = [ np.where(labels==lab)[0] for lab in np.unique(labels) ]

        # Now use similar code to k-fold, portioning to the subsets.
        subsets = [ [] for i in range(nfolds) ]

        for ec in equivclasses:
            nec = len(ec)
            ecshuff = np.random.permutation(ec)

            # This implementation isn't perfect for stratified case, but
            # good enough with large (few hundred) data points.
            # Issue is that the element aren't as evenly distributed
            # as they could be due to the nature of np.linspace.
            idxs = np.linspace(0,nec,nfolds+1).astype(int)
            for i in range(nfolds):
                il,ir = idxs[i],idxs[i+1]
                subsets[i].append(ecshuff[il:ir])
            #
        #
        for i in range(nfolds):
            subsets[i] = np.concatenate(subsets[i])
            # Reshuffle to remove ordering within subsets which could
            # introduce some form of bias.
            subsets[i] = np.random.permutation(subsets[i])
        #
    elif (method.startswith('split')):
        # input should be of the format "split-70:30"
        try:
            train_ratio,test_ratio = np.array(method.split('-')[1].split(':')).astype(int)
        except Exception as err:
            raise ValueError("Invalid string as cross-validation method")
            #import sys
            #sys.exit("\nInvalid string as cross-validation method\nExiting...")

        if(train_ratio + test_ratio != 100):
            raise ValueError("Split ratio of test and train sets should add up to 100")
            #import sys
            #sys.exit("\nSplit ratio of test and train sets should add up to 100\nExiting...")


        # Reusing the same methodology as the strtified_k-fold
        # Build a list of lists. Each of these lists points to all
        # locations in perm with the same label.
        equivclasses = [ np.where(labels==lab)[0] for lab in np.unique(labels) ]

        # Now use similar code to k-fold, portioning to the subsets.
        for _ in range(nfolds):
            subsets = [ [] for i in range(2) ]
            for ec in equivclasses:
                nec = len(ec)
                ecshuff = np.random.permutation(ec)

                # This implementation isn't perfect for stratified case, but
                # good enough with large (few hundred) data points.
                # Issue is that the element aren't as evenly distributed
                # as they could be due to the nature of np.linspace.
                idxs =  [0, int(nec*test_ratio/100.0), nec]
                for i in range(2):
                    il,ir = idxs[i],idxs[i+1]
                    subsets[i].append(ecshuff[il:ir])
                #


            #
            for i in range(2):
                subsets[i] = np.concatenate(subsets[i])
                # Reshuffle to remove ordering within subsets which could
                # introduce some form of bias.
                subsets[i] = np.random.permutation(subsets[i])

            trsubset = list(subsets)
            testsubset = trsubset.pop(0)
            trsubset = np.concatenate(trsubset)
            partitions.append([trsubset,testsubset])
        return partitions
    elif (method=='leave-one-out'):
        # The subsets are just singleton sets of each index.
        subsets = [ [i] for i in range(n) ]
        # nfolds needs to be fixed here if the input is wrong.
        nfolds = n
    elif (method=='leave-one-attrvalue-out'):
        if len(attrvalues)==0:
            raise ValueError("The argument `attrvalues` must specified for leave-one-attrvalue-out partitioning")
        #
        nfolds = len(np.unique(attrvalues))

        #group of indices by attribute value
        grouped_attrvals = [ np.where(attrvalues==val)[0] for val in np.unique(attrvalues) ]

        # Now use similar code to k-fold, portioning to the subsets.
        subsets = [ [] for i in range(nfolds) ]

        for i, group in enumerate(grouped_attrvals):
            groupshuff = np.random.permutation(group)
            # each subset contains all datapoints with a certain attribute value; variable fold/group size
            subsets[i].append(groupshuff)
        #
        #print(subsets)
        #print(subsets.pop(0))
        for i in range(nfolds):
            trsubset = list(subsets)
            testsubset = trsubset.pop(i)[0]
            t = []
            for s in trsubset:
                t+=s[0].tolist()
            trsubset = np.array(t)
            partitions.append([trsubset,testsubset])
        #
        return partitions 
    else:
        raise NotImplementedError("Cross validation type not supported");

    # Now loop over the subsets, defining testing
    # and training sets. Append them to the partitions.

    for i in range(nfolds):
        trsubset = list(subsets)
        testsubset = trsubset.pop(i)
        trsubset = np.concatenate(trsubset)
        partitions.append([trsubset,testsubset])

    #

    return partitions

    #return list(sklearn.model_selection.StratifiedKFold(n_splits=nfolds, random_state=None, shuffle=False).split(np.zeros(n),labels))

#
