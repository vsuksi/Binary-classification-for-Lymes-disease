def generate_synthetic_ccd1(**kwargs):
    '''
    Generate a basic dataset with only two attributes;
    mainly useful just to verify querying works as intended.
    '''
    import calcom
    import numpy as np

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

    ccd = calcom.io.CCDataSet()

    ccd.add_attrs(attrnames, attrdescrs=attrdescrs)
    ccd.add_datapoints(data, attrnames, metadata)
    ccd.add_variable_names(vnames)

    return ccd
#

#################################################################
#
#
#
#################################################################

def generate_synthetic_ccd2(**kwargs):
    '''
    Generates a dataset that roughly simulates
    what we see with human/animal challenge models
    which have multiple timepoints. Useful to test
    more complex cross-validations (one-subject-out,
    and potentially subject-k-fold).

    Here, there are forty subjects possess five time points each,
    and they have randomly assigned 'friendly' attributes by subject.
    '''
    import calcom
    import numpy as np

    seed = kwargs.get('seed', 2718281828)

    np.random.seed(seed=seed)

    nsubj = 40
    subjects = ['subject_%s'%str(i).zfill(2) for i in range(nsubj)]

    tpoints = [-1,0,1,2,3]
    nt = len(tpoints)

    n = nsubj*nt
    d = 5

    subjects = np.repeat(subjects, nt)
    times = np.tile(tpoints, nsubj)
    friendlies = np.hstack(
            [ np.repeat( np.random.choice([0,1]), nt) for _ in range(nsubj) ]
        )

    data = np.random.randn(n,d)

    # create separation by "friendly" attribute.
    data += np.sqrt(d)*np.dot( np.reshape(friendlies,(len(friendlies),1)), np.ones((1,d)) )

    attrnames = ['subject_id', 'time_id', 'friendly']

    metadata = [[subjects[i], times[i], friendlies[i]] for i in range(n)]

    ccd = calcom.io.CCDataSet()

    ccd.add_attrs(attrnames)
    ccd.add_datapoints(data, attrnames, metadata)
#    ccd.add_variable_names(vnames)

    return ccd
#
