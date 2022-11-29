from __future__ import absolute_import, division, print_function
from calcom.classifiers import AbstractClassifier
# import numpy as np
#from calcom.classifiers.nmf import helper_funs as hf

class NMFClassifier(AbstractClassifier):

    def __init__(self):
        '''
        Setup default parameters
        '''
        from numpy import random
        self.params = {}
        self.params['r'] = 2        # Number of classes
        self.params['nit'] = 1000    # Number of iterations used for the
                                    #   minimization procedure.

        # self.params['W0'] = random.rand(2,2)    # Initial W matrix
        # self.params['H0'] = random.rand(2,2)    # Initial H matrix

        self.params['save_diag'] = False    # Save diagnostics? (residual history, classification history)


        self.results = {}
        self.results['pred_labels']=None
        self.results['resid_hist'] = []      # History of residuals ||V-WH||_2
        self.results['class_hist'] = []      # History of classifications

        super().__init__() # Used to load parameters from file

    @property
    def _is_native_multiclass(self):
        return False
    #

    def fit(self,data,labels=None):
        '''

        Fit/model building function for non-negative matrix factorization.

        Syntax: fit(data,labels=None)

        Inputs:
            * data: n-by-m array of data, with n the
            number of observations; m, the dimensionality of observations.
            * labels:
                - If None (default), the labels on a call to predict()
            are assigned by the location of the basis vector in the array.
            For example, if we have basis vectors [u, v, w], then the
            labels are 0, 1, and 2, corresponding to u, v, w, respectively.
                - If provided, the clusters are given labels from numpy.unique(labels)
            that gives the best classification accuracy on the training data.
            For now, it is assumed that len(numpy.unique(labels)) = self.params['r'];
            if not, it is corrected internally.
            In this case, the overall algorithm is only quasi-unsupervised.


        On successful completion:
            * self.params['W'] and self.params['H'] are changed,
            * self.params['QR']
            * Diagnostics are saved if self.params['save_diag'] = True


        This algorithm interprets the input "data" as a
        matrix V, assumed to have non-negative entries,
        and decomposes it into a product of two
        non-negative matrices WH, where the inner dimension
        r will determine the number of classes (using the
        max along rows of W as the classifier).

        The classification is done using the interpretation
        V_i ~= sum( W_ij * H_j ), over rows of H, which is closest approximated
        (in some sense) by argmax(W_ij), over indices j.

        Parameters:

            r: number of classes desired (this is fixed; not adaptive) (default: 2)
            nit: number of iterations for minimization procedure (default: 100)
            W0: Initial guess for W. (default: random 2x2 array)
            H0: Initial guess for H. (default: random 2x2 array)

        Note that checks are done on W0 and H0 to ensure they have
        compatible dimensions. If they don't, new random matrices
        with compatible dimensions are created.

        '''
        import numpy as np
        #from numpy import zeros,shape,random,argmax

        internal_labels = self._process_input_labels(labels)

        # Unpack the parameters
        r = self.params['r']
        nit = self.params['nit']
        # W0 = self.params['W0']
        # H0 = self.params['H0']

        # If labels are given, reset the number of classes to
        # the number of unique labels.
        nonetype = type(None)
        if type(labels)!=nonetype:
            r = len(self._label_info['unique_labels'])
            self.params['r'] = r
        #


        # Check if W0 and H0 have the proper sizes; if not,
        # replace with random initial guesses.

        n,m = np.shape(data)

        # if not (np.shape(W0) == (n,r) and np.shape(H0) == (r,m) ):
        W = np.random.rand(n,r)
        H = np.random.rand(r,m)

        # For now, copy the data and feed the columnwise format
        # to the algorithm. The "roles" of W and H flip as well.
        datacopy = data.T
        Wcopy = np.array(W)
        W = H.T
        H = Wcopy.T

        # Run the algorithm
        nmf_algo(datacopy,W,H,nit)

        # Find a QR decomposition of W and store it.
        self.results['QR'] = np.linalg.qr(W, mode='reduced')

        # If label information is given, find a mapping of {0,...,r-1} to
        # the label set that maximizes classification accuracy on the
        # training set. This is nontrivial in general, so for now
        # we just use a heuristic here.
        # JUST KIDDING
        # NOT DONE
        lm = range(r)
        # if type(labels)!=nonetype:
        #     charclasses = []
        #     for i in range(r):
        #         charclasses.append()
        #
        # self.params['labelmap'] = lm
    #

    def predict(self,data):
        '''
        Classification step.

        Given an orthonormal basis of W, calculate the coefficients c_i in the basis.
        For each vector, classification is done using argmax(abs(c_i)).

        The labels are then mapped using self.params['labelmap'].
        Currently this isn't implemented; ony the identity map is used.
        So labels are defaulted to {0,...,r-1}.

        '''
        import numpy as np
        # Classify.
        n,m = np.shape(data)
        labeled_data = np.zeros(n).astype(int)
        Q,R = self.results['QR']

        datacopy = data.T
        dotprod = np.dot(Q.T,datacopy)
        projections = np.linalg.solve( R, dotprod )

        for i in range(n):
            labeled_data[i] = np.argmax(projections[:,i])
        #

        labeled_data = labeled_data.astype(int)

        pred_labels = self._process_output_labels(labeled_data)
        # self.results['pred_labels']=labeled_data

        return pred_labels


    # end classify

    def visualize(self,data,dim=2):
        '''

        Visualization function. Clustering is done using self.params['r']
        clusters, but visualization is done by projecting the r-by-n array proj_data
        reduced array into dim-by-n using the dimensions which have the
        greatest one-norm variation from their mean values.

        '''
        import numpy as np
        #self.params['r'] = max(self.params['r'],dim)
        #self.fit(data)

        # Get clusters and project the data.
        # Note this does a lot of the same work that self.predict() does.
        labels = self.predict(data)

        Q,R = self.params['QR']

        datacopy = data.T
        dotprod = np.dot(Q.T,datacopy)

        projections = np.linalg.solve( R, dotprod )
        proj_data = projections.T

        # Indexes of "best" dimensions to visualize with.
        # For now let's look at one-norm variation from the means.
        means = np.mean(proj_data,axis=0)
        merp = np.array( [ row - means for row in proj_data ] )
        merp = np.linalg.norm(merp, ord=1, axis=0)
        di = np.sort( np.argsort( merp )[-dim:] )

        import calcom.plot_wrapper as plotter
        if (dim==2):
            plotter.scatter(proj_data[:,di[0]],proj_data[:,di[1]],labels,title="NMF Classifier")
        else:
            plotter.scatter3d(proj_data[:,di[0]],proj_data[:,di[1]],proj_data[:,di[2]],labels,title="NMF Classifier")
        #


    # end visualize

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

# end class NMFClassifier

# The actual NMF algorithm is below.

def nmf_algo(V,W,H,niter):
    '''
    This is the algorithm proper. No checking of inputs is done.

    V is unchanged.
    W,H are changed on output.

    The outputs W,H are chosen such that the columns of W are 2-normalized,
    for whatever good it does. This is equivalent to multiplying by
    a diagonal matrix; (W*D)(D^-1*H) which gives the same approximation
    to V.

    '''

    from numpy import shape
    from numpy.linalg import norm

    r,_ = shape(H)

    for i in range(niter):
        nmf_iter(V,W,H)

    # Rescale the final W and H so that the columns of W are
    # all norm one.
    # Note the product WH is unchanged.
    for i in range(r):
        colnorm = norm(W[:,i])
        W[:,i] /= colnorm
        H[i,:] *= colnorm

# end nmf_algo


def nmf_iter(V,W,H):
    '''
    Does a (pair) of iterations for non-negative matrix factorization.
    Does no error checking whatsoever. This function shouldn't be
    called on its own unless you have a really good reason.

    V is not touched.
    W and H are changed on completion.
    '''

    from numpy import dot,shape

    n,r = shape(W)
    _,m = shape(H)

    # First: H_ij = H_ij*(W^T*V)_ij/(W^T*W*H)_ij

    numer = dot(W.T,V)
    denom = dot(dot(W.T,W),H)

    for i in range(r):
        for j in range(m):
            H[i,j] *= numer[i,j]/denom[i,j]


    # Second: W_ij = W_ij*(V*H^T)_ij/(W*H*H^T)_ij

    numer = dot(V,H.T)
    denom = dot(dot(W,H),H.T)

    for i in range(n):
        for j in range(r):
            W[i,j] *= numer[i,j]/denom[i,j]

# end nmf_iter
