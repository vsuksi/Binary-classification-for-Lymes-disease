def nnbp_generator(g_net, d_net):
    '''
    Inputs:
        g_net   - an object
        d_net   - an object
    Outputs:
        g_net   - an object.
    '''
    import numpy as np

    n = g_net.n
    g_o = g_net.layers[-1].a

    g_net.layers[-1].d = d_net.layers[0].d * g_o*(1-g_o)

    for i in range(n-2,-1,-1):  # n-2 to 0
        d = g_net.layers[i+1].d
        w = g_net.layers[i].w
        if i>0:
            a = g_net.layers[i].a
            g_net.layers[i].d = np.dot(d,w.T) * (a*(1-a))
        else:
            g_net.layers[i].d = np.dot(d,w.T)
        #
    #
    for i in range(n-1):    # 0 to n-2
        d = g_net.layers[i+1].d
        a = g_net.layers[i].a
        g_net.layers[i].dw = np.dot(a.T,d) / np.shape(d)[0]
        g_net.layers[i].db = np.nanmean(d, axis=0)  # Will this get me in trouble at some point?
    #

    return g_net
#
