from __future__ import absolute_import, division, print_function
from calcom.visualizers._abstractvisualizer import AbstractVisualizer

class PCAVisualizer(AbstractVisualizer):

    def __init__(self):
        '''
        Setup default parameters (a dictionary). Defaults are:

        params['d'] = 2
        params['eigensolver'] = np.linalg.eigh

        'd' is the number of dimensions for the PCA.
        'eigensolver' is a method which takes in a nonnegative definite
        square matrix and returns the eigenvalues and eigenvectors.
        Only "full" eigensolvers are implemented right now;
        so, no eigs or similar Krylov-based methods.

        '''
        from numpy.linalg import eigh
        self.params = {}
        self.params['dim'] = 2                      # Number of dimensions (2 or 3 only)
        self.params['eigensolver'] = eigh           # Possibly scipy.sparse.linalg.eigs?

        self.results = {}

        # TODO: possibly take in solver parameters, use eigs as default?
    #

    def project(self,data, **kwargs):
        '''
        Inputs:
            data: data array, n-by-m, where n is the number of observations
                and m is the dimensionality of the data.

        Optional inputs:
            dim: dimensionality of projection; overwrites self.params['dim']
            mean_center: if True, centers data by the mean of the rows
                internally before any decompositions are done. (Default: True)

        Output: coordinate array, n-by-d, where d is the number of dimensions
            of the projection.

        NOTE: This implementation is very inefficient for the moment,
            as the full eigendecomposition of the covariance matrix is calculated.
            Need to investigate where the ARPACK interface is to calculate
            only the top eigenpairs.
        '''
        import numpy as np

        self.params['dim'] = kwargs.get('dim', self.params['dim'])

        d = self.params['dim']

        data = np.array(data)

        data_mean = np.mean(data,axis=0)

        if kwargs.get('mean_center', True):
            data -= data_mean
        #

        # Based on the size of the matrix,
        # form the product appropriate to the expected
        # rank of the thing.
        #
        # From now on we're working in column-major format
        # for our own sanity.
        data = data.T
        m,n = data.shape
        if m<n:
            data_sq = np.dot(data, data.T)

            w,U_singvec = self.params['eigensolver'](data_sq)
            idxs = np.argsort(-w)

            U_singvec = U_singvec[:,idxs[:d]]
        else:
            data_sq = np.dot(data.T, data)

            w,V_singvec = self.params['eigensolver'](data_sq)
            idxs = np.argsort(-w)

            V_singvec = V_singvec[:,idxs[:d]]
            U_singvec = np.dot(data, V_singvec)
            U_singvec /= np.linalg.norm(U_singvec,axis=0)
        #

        # Reconstruct coordinates in the reduced basis and return them.
        # remember: data is arragned in columns now.
        coords = np.dot(U_singvec.T,data).T
#        coords = (coords/np.sqrt( w[idxs[:d]] ))    # NOT SAFE WITH ZERO EIGENVALUES

        # Save scaled singular vectors for future projections.
        self.results['components'] = U_singvec
        self.results['singular_values'] = np.sqrt(w[idxs[:d]])
        self.results['data_mean'] = data_mean

        return coords
    #

    # def visualize(self,coords):
    #     '''
    #     Input: n by d array of (projected) data.
    #     '''
    #     d = self.params['dim']
    #     if (d==2 or d==3):
    #         plotter.scatterTrainTest(coords,self.labels,None,None,readable_label_map=self.readable_label_map,title="PCA Visualizer", dim=d)
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
