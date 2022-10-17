#from __future__ import absolute_import, division, print_function
# import numpy as np
# import scipy as sp

from calcom.classifiers._abstractclassifier import AbstractClassifier

class ABSClassifier(AbstractClassifier):
    '''
    Description here soon
    '''
    def __init__(self):

        # initialize parameters here
        self.params = {}
        self.params['dim'] = 5
        self.params['verbosity'] = 1
        self.params['metric'] = 'Euclid'

        self.results = {}
        self.results['pred_labels'] = []

        # self.subspaces = []
        # self.subspace_labels = []
        self.pairwise_dist = []

    @property
    def _is_native_multiclass(self):
        return True

    @property
    def _is_ensemble_method(self):
        return False

    def initParams(self, dim, verbosity, metric):
        '''
        Used to set all the input parameters in a one-liner.
        '''
        self.params['dim'] = dim
        self.params['verbosity'] = verbosity
        self.params['metric'] = metric

    def _fit(self, trData, trLabels):
        '''
        Fit function for Angles Between Subspaces classifier.

        Inputs:
            trData : data matrix to fit the model; a nupy array of shape
                n-by-d, where d is the dimensionality of the data.
            trLabels : corresponding labels for each row of the data, shape n.
        Outputs:
            None
        '''
        import numpy as np
        import scipy as sp

        # internal_labels = self._process_input_labels(trLabels)
        internal_labels = np.array(trLabels)

        trData = trData.T  # need column major
        classes = self._label_info['unique_labels_mapped']

        lbl = []
        subspaces = []
        for i in classes:
            class_data = trData[:, np.where(internal_labels == i)[0]]
            _, n = class_data.shape
            subspace_count = n // self.params['dim']
            r = n % self.params['dim']

            if self.params['dim'] == 1:
                for j in range(subspace_count):
                    subspaces.append(class_data[:, j] / np.linalg.norm(class_data[:, j]))
                    lbl.append(i)
            else:
                for j in range(subspace_count):
                    indx = self.knn(self.params['dim'] - 1, class_data)
                    subspace = sp.linalg.orth(class_data[:, indx[:, 1]])
                    subspaces.append(subspace)
                    lbl.append(i)
                    np.delete(class_data, indx[:, 1], axis=1)
                if r != 0:
                    subspaces.append(sp.linalg.orth(class_data))
                    lbl.append(i)
        #
        # self.subspaces = subspaces
        # self.subspace_labels = lbl

        self.results['subspaces'] = subspaces
        self.results['subspace_labels'] = lbl

        return
    #

    def _predict(self, data):
        '''
        Returns predicted labels for a set of input data, after
        the self.fit() function has been called with training
        data and labels.

        Inputs:
            data : numpy array dimension n-by-d; where d matches
                the dimensionality of the data used in training
        Outputs:
            pred_labels : predicted labels
        '''
        import numpy as np

        nsampl, dim = data.shape

        # nsubspace = len(self.subspaces)
        nsubspace = len(self.results['subspaces'])

        thetaindx = []

        for i in range(nsampl):
            thetas = []
            tdata = np.array(data[i, :])
            tdata = tdata / np.linalg.norm(tdata)

            for j in range(nsubspace):
                # subdata = self.subspaces[j]
                subdata = self.results['subspaces'][j]
                F = np.amax(np.absolute(np.matmul(tdata.T, subdata)))
                thetas.append(F)

            indx = np.argmax(thetas)
            thetaindx.append(indx)

        thetalbl = []
        for i in range(nsampl):
            thetalbl.append(self.results['subspace_labels'][thetaindx[i]])
        #

        # self.results['pred_labels'] = thetalbl
        # pred_labels = self._process_output_labels(thetalbl)

        # return pred_labels
        return thetalbl
    #

    def visualize(self, *args):
        pass

    def knn(self, k, data):
        import numpy as np

        # TODO: add option to switch between euclidean and geodesic
        _, n = data.shape
        x2 = np.sum(np.square(data), axis=0)
        dist = np.tile(x2, (n, 1)) + np.tile(x2.T, (n, 1)) - 2 * np.matmul(data.T, data)
        index = np.argsort(dist, axis=0)
        if k >= n:
            idx = index
        else:
            idx = index[1:k + 1, ]

        self.pairwise_dist = dist

        return idx


    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
