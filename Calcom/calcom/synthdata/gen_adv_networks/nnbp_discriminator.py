def nnbp_discriminator(nn, y_h, y):
    '''
    Inputs:
        nn  - an object
        y_h -
        y   -
    Outputs:
        nn  - the object, but updated.
    '''
    import numpy as np

    n = nn.n
    nn.layers[n-1].d = -((y/y_h) + (y-1)/(1.-y_h))*(y_h*(1.-y_h))
    for i in range(n-2,-1,-1):  # n-2 to 0
        d = nn.layers[i+1].d
        w = nn.layers[i].w

        if i>0:
            a = nn.layers[i].a
            nn.layers[i].d = np.dot(d,w.T)*a*(1-a)
        else:
            nn.layers[i].d = np.dot(d,w.T)
        #
    #

    for i in range(n-1):    # 0 to n-2
        d = nn.layers[i+1].d
        a = nn.layers[i].a

        nn.layers[i].dw = np.dot(a.T,d)/(np.shape(d)[0])
        nn.layers[i].db = np.nanmean(d, axis=0) # Will this get me in trouble at some point?
    #
    return nn
#
