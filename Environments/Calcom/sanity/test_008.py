def test(**kwargs):
    '''
    Test 008: Is it possible to make a basic CCDataSet and
    perform queries on it?
    '''
    import calcom
    import numpy as np

    # Note: this first section is a rough duplicate
    # of the synthetic dataset created in
    # calcom.utils.synthetic_datasets (as of April 15, 2019)

    subtests = []

    seed = kwargs.get('seed', 2718281828)

    np.random.seed(seed=seed)

    n0,d = 100,4

    n = 2*n0

    attr0 = np.random.choice([0,1], n)
    attr1 = np.hstack([ np.repeat('lion',n0), np.repeat('tiger', n0) ])
    data = np.random.randn(n,d)

    # Create a separation in the data based on the type of animal
    data[:n0] += np.sqrt(d)

    attrnames = ['friendly', 'animal']
    attrdescrs = ['Whether this particular animal is friendly or not (1=yes)',
        'The type of animal'
    ]

    metadata = [[attr0[i],attr1[i]] for i in range(n)]

    vnames = ['x%i'%i for i in range(d)]

    try:
        # Can we successfully create the dataset?
        ccd = calcom.io.CCDataSet()

        ccd.add_attrs(attrnames, attrdescrs=attrdescrs)
        ccd.add_datapoints(data, attrnames, metadata)
        ccd.add_variable_names(vnames)

        # nothing fancy yet - would be a problem if
        # we got an error at this point.
        subtests.append( True )
    except:
        subtests.append( False )
    #

    # Now test out some basic functionality.

    # does partitioning construct an equivalence class
    # on the requested metadata?
    try:
        eq0 = ccd.partition('friendly')
        blerp = list(eq0.keys())
        blerp.sort()
        subtests.append( blerp==[0,1] )

        for j in [0,1]:
            subtests.append( all( np.array( ccd.get_attrs('friendly',idx=eq0[j]) )==j ) )
        #

        # repeat for the second attribute.
        eq1 = ccd.partition('animal')
        blerp = list(eq1.keys())
        blerp.sort()
        subtests.append( blerp==['lion','tiger'] )

        for j in ['lion','tiger']:
            subtests.append( all( np.array(ccd.get_attrs('animal',idx=eq1[j]))==j ) )
        #
    except:
        subtests.append( False )
    #

    try:
        #
        # Does basic searching behave as expected?
        # Everything satisfying a condition should be returned in
        # the list of pointers; everything in the complement shouldn't
        # satisfy at least one condition.
        query = {'animal':'tiger', 'friendly':0}

        idx = ccd.find(query)
        complement = np.setdiff1d( list(range(n)), idx )

        selected_a0 = ccd.get_attrs('friendly', idx=idx)
        selected_a0 = np.array(selected_a0)
        selected_a1 = ccd.get_attrs('animal', idx=idx)
        selected_a1 = np.array(selected_a1)

        satisfied = np.logical_and(
                selected_a0==query['friendly'],
                selected_a1==query['animal']
                )

        subtests.append( all(satisfied) )

        excluded_a0 = ccd.get_attrs('friendly', idx=complement)
        excluded_a0 = np.array(excluded_a0)
        excluded_a1 = ccd.get_attrs('animal', idx=complement)
        excluded_a1 = np.array(excluded_a1)

        properly_excluded = np.logical_or(
                excluded_a0 != query['friendly'],
                excluded_a1 != query['animal']
        )

        subtests.append( all(properly_excluded) )

    except:
        subtests.append( False )
    #

    return all( subtests )
#

if __name__=="__main__":
    print( test() )
