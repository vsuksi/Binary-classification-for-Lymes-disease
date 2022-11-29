from __future__ import absolute_import, division, print_function
from calcom.visualizers._abstractvisualizer import AbstractVisualizer
# import calcom.plot_wrapper as plotter

class LTSAVisualizer(AbstractVisualizer):
    def __init__(self):
        '''
        initial parameters for LTSA. Defaults are:

        params['dim'] = 2
        params['nn'] = 12

        'd' is plot dimension for LTSA
        'nn' is the number of nearest neighbors chosen to build the tangent space at each point
        '''

        self.params = {}
        self.params['dim'] = 2
        self.params['nn'] = 12
        self.params['eps'] = 1e-14

    def ltsa(self,data):
        '''
        data: data array, n x m, where n is the # of obeservations and m is the number of varaibles
        '''
        from numpy.linalg import svd, eig
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        X = data.T
        k = self.params['nn']
        d = self.params['dim']
        m,N = np.shape(X)
        e = np.ones((k,1))

        # step 1: Extracting loval information
        # find k nearest neighbor
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X.T)
        distances, idx = nbrs.kneighbors(X.T)
        G = {}

        # compute the largest eigenvectors of Xo'Xo
        for i in range(0,N):
            Xi = X[:,idx[i,:]]
            Xi_mean = Xi.mean(axis=1)
            #pdb.set_trace()
            Xi_o = Xi - Xi_mean[:,np.newaxis]
            g,_,_ = svd((Xi_o.T).dot(Xi_o),full_matrices = False);
            G[i] = np.hstack((e/(k**(0.5)),g[:,:d]))

        # construct alignment matrix
        B = np.zeros((N,N))
        for i in range(0,N):
            #pdb.set_trace()
            B[np.ix_(idx[i,:],idx[i,:])] = B[np.ix_(idx[i,:],idx[i,:])] + np.eye(k) - G[i].dot(G[i].T)
            #print(B)

        S,U = eig(B)
        ind = np.argsort(S)
        T = U[:,ind]
        feature = T[:,1:d+1]
        #pdb.set_trace()

        return feature


    # def project(self, data, labels, readable_label_map ={}, nn=None, dim=None):
    def project(self,data, **kwargs):
        '''
        Inputs:
            data: data array, n x m, where n is the number of observations and m is the number of variables.

        Optional inputs:
            dim: integer; dimensionality of projection. If specified, overwrites self.params['dim']
            nn: integer; number of nearest neighbors chosen to build the tangent space at each point.
                If specified, overwrites self.params['nn']

        Outputs:
            coords: numpy array, n x d, where d is the number of dimensions of the projection
        '''
        # overide default dim and nn if dim or nn is specified
        self.params['dim'] = kwargs.get('dim', self.params['dim'])
        self.params['nn'] = kwargs.get('nn', self.params['nn'])

        coords = self.ltsa(data)

        return coords
    #

    # def visualize(self,coords):
    #     '''
    #     Input: n by d array of (projected) data.
    #     '''
    #     d = self.params['dim']
    #     if (d==2 or d==3):
    #         plotter.scatterTrainTest(coords,self.labels,None,None,readable_label_map=self.readable_label_map,title="LTSA Visualizer", dim=d)
    #     else:
    #         print('Error: only d=2 and d=3 are supported for scatterplotting PCA')
    #         return None,None
    # #
    def visualize(self,coords,labels=None,label_map=None):
        '''
        Inputs:
            coords: n-by-d array of projected coordinates, with d=2 or 3.
                Automatically handles making a 2d/3d plot based on the value of d.
        Optional inputs:
            labels: (n,) array of labels.
            label_map: dictionary mapping labels to plaintext descriptions.
                If not specified, the values in labels are used.
            show: Boolean. Whether to show the plot immediately. (Default: True)
        Outputs:
            fig,ax: pyplot figure/axis pair of the resulting plot, as generated
                from fig,ax = pyplot.subplots(1,1).
        '''

        raise NotImplementedError('Visualization tools under renovation.')
        return
    #
#
