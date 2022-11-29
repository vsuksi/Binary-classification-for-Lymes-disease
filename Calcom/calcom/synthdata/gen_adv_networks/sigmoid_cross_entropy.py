def sigmoid_cross_entropy(y_h, y):
    '''
    Returns the cross entropy function for GAN.
    '''
    import numpy as np
    result = -(y*np.log(y_h) + (1-y)*np.log(1-y_h))
    return np.nanmean(result)   # Will this get me in trouble some day?
#
