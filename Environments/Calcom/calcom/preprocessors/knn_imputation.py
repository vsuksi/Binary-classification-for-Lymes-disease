from __future__ import absolute_import, division, print_function
from calcom.preprocessors import AbstractPreprocessor


class KNNImputation(AbstractPreprocessor):

    def __init__(self):
        '''
        Setup default parameters

        For k-nearest-neighbors imputation:
            self.params['k'] : Integer, specifying number of neighbors to look at. (default: 3)
            self.params['labels'] : Optional list of labels. If specified, points in different
                                    classes are treated as having infinite distance.
            self.params['weight'] : String indicating weighting to use for nearest neighbors.
                                        'equal' : All neighbors have equal weight
                                        'inv_dist' : Weight is inversely proportional to the distance.
        '''
        self.params = {}
        self.params['k'] = 3
        self.params['labels'] = []
        self.params['weight'] = 'equal'

        self.results = {}
        self.results['distances'] = []
        self.results['distmat'] = []
        self.results['neighbors'] = []

    #

    def process(self,data):
        '''
        Imputes missing values in the data array by taking the mean
        of the values of its k nearest neighbors. The distance for each
        pair of points is the standard Euclidean norm, scaled by
        the dimension of their common subspace.

        Inputs:

        data:       n-by-r array of data, with n the number of observations.

        Parameters:

        self.params['k']:

                    Integer, number of neighbors to look at. Default: 3

        self.params['labels']:

                    Labels for each data point. If specified, nearest neighbors
                    are restricted within classes; e.g., a data labeled 0 will
                    only look to neighbors whose label is also 0 to impute
                    a missing value. (This is implemented sloppily, a proper
                    version may come eventually).

        Outputs:

        data:       Modified dataset of the same dimensions with imputed data values.

        '''

        import numpy as np

        n,r = np.shape(data)

        # If labels aren't specified, use a fictitious label for all data points.
        if not len(self.params['labels']):
            labels = np.zeros(n)
            # unique_labels = np.array([0])
        else:
            labels = self.params['labels']
            # unique_labels = np.unique(labels)
        #

        # Generate distance matrix. Labels in different classes are assigned
        # infinite distance (easiest implementation for now, better implementation
        # is to generate a distance matrix for each label subset)

        dists = {}
        nns = {}
        for i in range(n):
            dists[i] = []
            for j in range(n):  # Yes, I know, but I'm lazy, and O(r**2) is O(r**2) regardless.
                if (labels[i]==labels[j]):
                    # Concatenating lists, not adding values.
                    val = vec_impute_distance(data[i,:],data[j,:])
                    dists[i] = dists[i] + [ val ]
                else:
                    # distmat[i,j] = np.inf
                    dists[i] = dists[i] + [ np.inf ]
                #
            #
            idxs = list( np.argsort(dists[i]) )


            # Would like the point itself to always lead the list.
            idxs.remove(i)
            idxs = [i] + idxs
            derp = list(dists[i])

            # dists[i] = derp[:i] + derp[i+1:]


            nns[i] = idxs
        #

        # Now find all the nan values, and take appropriate action.
        nani,nanj = np.where(data!=data)
        nnans = len(nani)

        dataout = np.array(data)

        for q in range(nnans):
            nn_q = np.array( nns[nani[q]] )
            nanmask = np.array( [ data[k,nanj[q]]==data[k,nanj[q]] for k in nn_q] )

            infmask = np.array( [ val<np.inf for val in dists[nani[q]] ] )


            valid_idxs, = np.where( nanmask * infmask )

            if len(valid_idxs)==0:
                print("Error: data[%i,%i] has no valid neighbors to impute data from.")
                k = 0
            elif len(valid_idxs)<self.params['k']:
                print("Warning: data[%i,%i] only has %i (<%i) values to impute data from."%(nani[q],nanj[q],len(valid_idxs),self.params['k']) )
                k = len(valid_idxs)
            else:
                k = self.params['k']
            #

            othervals = data[nn_q[valid_idxs[:k]],nanj[q]]

            if k==0:
                dataout[nani[q],nanj[q]] = dataout[nani[q],nanj[q]] # No change
            else:
                if self.params['weight'] == 'inv_distance':

                    dists_q = np.array( dists[nani[q]] )
                    idxs_q = np.array( valid_idxs[:k], dtype='int' )
                    blah = dists_q[idxs_q]

                    # Check for zero distance.
                    if any(blah==0):
                        weights = np.zeros(len(blah))
                        weights[blah==0] = 1.
                    else:
                        weights = 1./np.array( dists_q[idxs_q] )
                    #
                    weights /= np.sum(weights)

                elif self.params['weight'] == 'equal':
                    weights = np.ones(k)
                    weights /= np.sum(weights)
                else:
                    weights = np.ones(k)
                    weights /= np.sum(weights)
                #

                dataout[nani[q],nanj[q]] = np.dot(weights,othervals)
            #

        #

        self.results['distances'] = dists
        self.results['distmat'] = np.array( [dists[i] for i in range(len(dists))] )
        self.results['neighbors'] = nns

        data = dataout

        return data
    #
#

def vec_impute_distance(u,v):
    '''
    Calculates a distance d(u,v) between vectors u and v.
    The vectors u and/or v may have other
    missing values. The norm is calculated in the largest
    subspace for which both values are known, and normalized
    by the dimension of that subspace.

    If u and v share no common subspace, np.inf is returned.
    '''
    import numpy as np

    idxs1 = np.where(u==u)
    idxs2 = np.where(v==v)
    idxs = np.intersect1d(idxs1,idxs2)

    if len(idxs):
        return np.linalg.norm(u[idxs] - v[idxs])/len(idxs)
    else:
        return np.inf
    #
#

if __name__=="__main__":
    import numpy as np
    import calcom

    knn = calcom.preprocessors.KNNImputation()
    knn.params['k'] = 2
    knn.params['weight'] = 'inv_distance'
    a = np.array([[-1,np.nan,3,4,5],[3,5,4,4,np.nan],[3,1,3,4,6],[3,2,4,4,7],[1,4.5,6,4,5],[np.nan,4,np.nan,np.nan,np.nan]])

    a_ref = np.array(a)

    a = knn.process(a)

    print(a_ref)
    print(a)
#
