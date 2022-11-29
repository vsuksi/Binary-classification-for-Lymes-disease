if __name__ == "__main__":
    '''
    A demonstration that PCA fails to separate
    data coming from two classes, if they are lying
    on separate low-dimensional subspaces.
    '''

    import numpy as np
    from matplotlib import pyplot
    import calcom

    # Function to generate a random data matrix.
    def randomranktwo(n,m,eps=1e-2,sigmas=np.array([4.,1.])):
        from numpy.linalg import norm,qr
        U = np.zeros( (n,2) )
        for i in range(n//2):
            U[i,:] = np.array([1,0])
            U[i+n//2,:] = np.array([0,1])
        #

        U /= np.sqrt(n/2.)

        V,_ = qr(np.random.randn(m,2))

        # Some extra ugliness is needed to construct A using a sum of
        # rank-one matrices.
        noise = eps*min(sigmas)*np.random.randn(n,m)

        data = noise
        for i in range(2):
            ui = U[:,i]
            vi = V[:,i]
            ui.shape = (n,1)
            vi.shape = (m,1)
            data += sigmas[i] * np.dot(ui,vi.T)

        return data
    #

    ############################
    #
    # Set parameters.
    #

    n,m = 400,20
    eps = 1e-2

    # Create a random tall and skinny matrix which is rank-two plus noise.
    data = randomranktwo(n,m,eps)

    # The matrix is now made.
    # Now see how a PCA does at recovering the two-dimensional form of the thing.
    pcavis = calcom.visualizers.PCAVisualizer()

    coords = pcavis.project(data)
    pcavis.visualize(coords)
    #fig,ax = pcavis.visualize(coords, s=100, marker='*', color='yellow', edgecolor='black', linewidth=1)

    #ax.set_aspect('equal')
    #ax.grid()

    #pyplot.show(block=True)
