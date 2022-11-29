
def one_hot_dictionary(labels):
    '''
    Inputs:
        labels: a list-like of labels to be mapped.
    Outputs:
        mapping: a dictionary defining a map
            from a unique set of the original labels
            to R^(m-1), where m is the number of
            unique entries in labels

    '''
    import numpy as np
    mapping_int = {}
    # First pass: create a dictionary mapping
    # to integers.
    m = 0
    mapped = []
    for l in labels:
        if l not in mapping_int.keys():
            mapping_int[l] = m
            m += 1
        #
        mapped.append( mapping_int[l] )
    #

    mapping = {}
    z = np.zeros(m-1)
    for j,(k,v) in enumerate( mapping_int.items() ):
        mapping[k] = np.array(z)
        if j==m-1:
            mapping[k] = np.array(z)
            break
        else:
            mapping[k][j] = 1.
    #

    return mapping
#

def one_hot(labels,mapping):
    '''
    Returns the one-hot encoding of a
    list of labels under the given mapping, assumed
    generated from one_hot_dictionary.
    '''
    import numpy as np

    return np.array([mapping[l] for l in labels])
#
