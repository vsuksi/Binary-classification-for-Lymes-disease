#from __future__ import absolute_import, division, print_function
from calcom.metrics._abstractmetric import AbstractMetric
#from sklearn.metrics import accuracy_score

class DistanceMatrix(AbstractMetric):

    def __init__(self):
        '''
        Setup default parameters

        Parameters:
            'norm': Which norm to use to measure distance between pairs of rows.
            Defaults to 2.

            This argument is passed to the 'ord' argument of
            numpy.linalg.norm internally. Possible options are 1 (usual 1-norm),
            2 (2-norm), numpy.inf (infinity/max-norm). Other real positive
            numbers passed to numpy.linalg.norm use the generic l_p norm formula.
        '''
        self.params = {}
        self.params['norm'] = 2
    #

    def evaluate(self, data):
        '''
        Explicitly creates the symmetric distance matrix of the given data array,
        assumed to be shape d-by-n, with n data points in R^d. Currently
        only supports those norms implemented in numpy.linalg.norm() by the
        "ord" argument. Change the norm by modifying self.params['norm'].

        Inputs:
            data: 2d numpy array, with data.shape = (n,d), of n data points in R^d.

        Outputs:
            distmat: 2d numpy array, with distmat.shape = (n,n), where
            distmat[i,j] = np.linalg.norm(data[i,:] - data[j,:], ord=self.params['norm'])

        '''

        import numpy as np

        n,d = data.shape

        print(d,n)

        distmat = np.zeros( (n,n) )

        for i in range(n):
            for j in range(i+1,n):
                dist = np.linalg.norm(data[i,:] - data[j,:], ord=self.params['norm'])
                distmat[i,j] = dist
                distmat[j,i] = dist
            #
        #

        return distmat
    #
#
