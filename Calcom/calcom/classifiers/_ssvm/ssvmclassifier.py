#from __future__ import absolute_import, division, print_function
# import numpy as np

from calcom.classifiers._abstractclassifier import AbstractClassifier
# try:
#     import torch
# except ImportError:
#     torch=None

class SSVMClassifier(AbstractClassifier):
    '''

    A class implementing a sparse support vector machine (SSVM) algorithm,
    which solves the *sparse* (l1) support vector problem. The
    primal form of the optimization problem is (?)

    min ||w||_1 + C*sum(xi_j)

        s.t. y_j*(w'*x_j - b) <= 1-xi_j,

    for points x_j with corresponding labels y_j (assumed +/-1 in this formulation),
    optimizing for vector w (the ``weight" vector) and scalar b (the ``bias"),
    with a corresponding linear (affine) model function f(x) = w'*x - b which
    approximates the original data, and classifies points using
    sign(f(x)).

    The Python implementation of this code is by Tomojit Ghosh, which was
    originally an adaptation of a Matlab code by Sofya Chepushtanova and
    Michael Kirby. Many minor changes were done to change this code to
    fit the template required by Calcom.

    '''

    def __init__(self):
        '''

        Initialization of default parameters, and some initial flags.

        self.results['weight']=np.array([])
        self.results['bias']=None

        # By default, output labels are +/-1.
        # If true, replaces -1 with 0 on a self.classify() call.
        # Internal algorithm is unaffected.
        self.params['use01Labels'] = False

        '''
        from calcom.solvers import LPPrimalDualPy

        self.params = {}

        # Solver parameters
        self.params['C']=1.0                    # the margin weight
        self.params['TOL']=0.001                # error tolerance for interior point method
        self.params['method']=LPPrimalDualPy    # method for solving the LP
        self.params['errorTrace']=None
        self.params['use_cuda'] = False          # Flag to attempt to use CUDA.
        self.params['verbosity'] = 0            # Level of verbosity
        # self.params['inputDim']=0
        self.params['w'] = None
        self.params['debug'] = False

        # By default, output labels are +/-1.
        # If true, replaces -1 with 0 on a self.classify() call.
        # Internal algorithm is unaffected.

        # SOON TO BE DEPRECATED
        # self.params['use01Labels'] = True

        # Model coefficients and labels
        self.results = {}
        # self.results['weight']=np.array([])
        self.results['weight']=[]
        self.results['bias']=None
        self.results['pred_labels']=[]

        super().__init__()

    #

    # def initParams(self,c,tol,method,errorTrace,inputDim):
    def initParams(self,c,tol,method,errorTrace,inputDim,use_cuda,verbosity):
        '''
        Used to set all the input parameters in a one-liner.
        '''
        self.params['C']=c              #the margin weight
        self.params['TOL']=tol          #error tolerance for interior point method
        self.params['method']=method    #method for solving the LP
        self.params['errorTrace']=errorTrace
        self.params['use_cuda'] = use_cuda
        self.params['verbosity'] = verbosity
        # self.params['inputDim']=inputDim

    #

    @property
    def _is_native_multiclass(self):
        '''
        Must return True or False
        '''
        return False
    #
    @property
    def _is_ensemble_method(self):
        return False

    def _fit(self,trData,trLabels):
        '''
        Fit/training step for Sparse Support Vector Machines (SSVM). A model function

        f(x)=w'*x - b

        is found for vector w=len(x) and scalar b which optimally classify the
        training data, in the sense of solving the L1 minimization problem

        min ||w||_1 + C*sum( c_j * xi_j )
            s.t. y_j*(w'*x_j -b) <= 1-xi_j, j=1,...,n,

        where c_j are positive and sum to one, x_j are vector input data, y_j are {-1,+1} class labels for each x_j,
        xi_j are scalar slack variables.

        This code only supports binary classification right now. If the
        training labels are {0,1}, then self.params['use01Labels'] is set to True, and
        the 0 labels are internally converted to -1.

        The weight vector w and bias b are stored in self.results['weight'] and
        self.results['bias'] respectively.

        '''
        import numpy as np
        try:
            import torch
        except ImportError:
            torch=None

#        internal_labels = self._process_input_labels(trLabels)

        # Check that the stars have aligned so that we can use CUDA.
        use_cuda = self.params['use_cuda'] and torch and torch.cuda.is_available()
        if self.params['verbosity']>0:
            if self.params['use_cuda'] and not use_cuda:
                print('PyTorch could not be imported, or could not access the GPU. Falling back to numpy implementation.')
        #

        if ( len(self._label_info['unique_labels_mapped']) != 2 ):
            raise ValueError("The supplied training data has fewer or greater than two labels.\nOnly binary classification is supported.")

        # Need an extra step here - SSVM wants labels -1 and 1.
        self._lmap2 = {0:-1, 1:1}
        self._ilmap2 = {-1:0, 1:1}
        internalLabels = [self._lmap2[l] for l in trLabels]

        nSamples = np.shape(trData)[0]

        inputDim=np.shape(trData)[1]
        IP=np.diag(np.ones(nSamples)).astype(int)
        eP=np.ones(nSamples).reshape(-1,1)

        if self.params['w'] is None:
            self.params['w'] = np.ones(nSamples).reshape(-1,1)
        else:
            self.params['w'] = nSamples * (self.params['w'] / np.sum(self.params['w'])).reshape(-1,1)

        eDim=np.ones(inputDim).reshape(-1,1)

        D=np.diag(internalLabels)    #Diagonal matrix of labels

        if use_cuda:
            D_c = torch.from_numpy(D).double().cuda()
            trData_c = torch.from_numpy(trData).double().cuda()
            DX = torch.mm(D_c,trData_c).cpu().numpy();

            eP_c = torch.from_numpy(eP).double().cuda()
            De = torch.mm(D_c,eP_c).cpu().numpy();
            #self.params['w'] = torch.from_numpy(self.params['w']).double().cuda()
        else:
            DX=np.dot(D,trData)
            De=np.dot(D,eP)
        #

        A = np.hstack((DX,-DX,-De,De,IP))
        c = np.vstack((eDim, eDim, np.array([0]).reshape(-1,1), np.array([0]).reshape(-1,1), self.params['C'] * self.params['w']))

        x = self.params['method'](-c,-A,-eP, output_flag=0, use_cuda=use_cuda, verbosity=self.params['verbosity'], debug=self.params['debug'])

        self.results['weight'] = x[:inputDim] - x[inputDim:2*inputDim]
        self.results['bias'] = x[2*inputDim]-x[2*inputDim+1]

        #return
        return self
    #

    def _predict(self,data):
        '''
        Classification step for Sparse Support Vector Machine (SSVM).
        After the fit/training step, vectors w and b are found to
        optimally classify the training data (in the sense described
        in the fit() docstring). New data is classified using

        sign(f(x)) = sign( w'*x - b ).

        If self.params['use01Labels'] = True, the -1 labels are replaced
        with 0 labels.
        '''
        import numpy as np
        try:
            import torch
        except ImportError:
            torch=None

        # Check that the stars have aligned so that we can use CUDA.
        use_cuda = self.params['use_cuda'] and torch and torch.cuda.is_available()
        if self.params['verbosity']>0:
            if self.params['use_cuda'] and not use_cuda:
                print('PyTorch could not be imported, or could not access the GPU. Falling back to numpy implementation.')
        #


        b = self.results['bias']
        w = self.results['weight']

        if use_cuda:
            data_c = torch.from_numpy(data).double().cuda()
            w_c = torch.from_numpy(w).double().cuda()
            b_c = torch.from_numpy(b).double().cuda()
            d = torch.addmm(-1,b_c,data_c,w_c).cpu().numpy()
        else:
            d = np.dot(data,w) - b

        predicted = np.sign(d)
        predicted = np.array(predicted, dtype=int).flatten()  # can't be too sure

        # Extra step here needed to map {-1,1} to {0,1}.
        pred_labels2 = [self._ilmap2[l] for l in predicted]

#        pred_labels = self._process_output_labels(pred_labels2)

#        return pred_labels
        return pred_labels2
    #

    def decision_function(self,data):
        import numpy as np
        try:
            import torch
        except:
            torch=None
        #

        b = self.results['bias']
        w = self.results['weight']
        if torch and self.params['use_cuda']:
            d = torch.addmm(-1,b_c,data_c,w_c).cpu().numpy()
        else:
            d = np.dot(data,w) - b
        #
        return d
    #

    def visualize(self,*args):
        pass

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
