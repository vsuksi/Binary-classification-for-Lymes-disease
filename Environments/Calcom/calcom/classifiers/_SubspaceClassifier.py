#from __future__ import absolute_import, division, print_function
from calcom.classifiers._abstractclassifier import AbstractClassifier

# TODO: fix this approach. "global torch" is used in a function below.
torch = None

class SubspaceClassifier(AbstractClassifier):
    def __init__(self):
        '''
        Setup subspace dimension and other default parameters
        '''
        self.params = {}
        self.params['dim'] = 5
        self.params['use_cuda'] = False # TODO: torch linear algebra ops in this code; especially in principal_angles.
        self.params['verbosity'] = 0

        self.results = {}
        self.results['pred_labels'] = []

        self.subspaces = []
        self.subspaces_label = []
        global torch
        try:
            import torch
        except ImportError:
            torch = None
    #

    @property
    def _is_native_multiclass(self):
        return True
    #
    @property
    def _is_ensemble_method(self):
        return False

    def _fit(self, data, labels):
        '''
        data: nxp matrix where n is number of samples and p and the number of variables
        labels: length n 1d array

        This function represents each class as a k dimensional subspace using provided labels
        '''
        import numpy as np

        'TODO: ADD a switch to centerting?'
        centered_data = data - np.matlib.repmat(np.mean(data,axis=0), data.shape[0], 1)

        # internal_labels = self._process_input_labels(labels)

        dims = []
        # internal_labels = np.array(internal_labels)
        labels = np.array(labels)
        for i in np.unique(labels):
            datai = data[np.where(internal_labels==i)[0],:]
            subspacei, ri = np.linalg.qr(datai.T)
            dimi = subspacei.shape[1]
            dims.append([dimi])
            self.subspaces.append(subspacei)
        if np.max(dims)>=self.params['dim']:
            if self.params['verbosity']>0:
                print('warning: specified parameter `dim` is larger than feasible dimension!\n Automatically selecting a feasible dimension')
            self.params['dim'] = np.min(dims)

        for i in range(0, len(self.subspaces)):
            self.subspaces[i] = self.subspaces[i][:,:self.params['dim']]
            self.subspaces_label.append(np.unique(internal_labels)[i])
    #

    def _predict(self, data):
        import numpy as np

        torch_available = torch and torch.cuda.is_available() and self.params['use_cuda']
        if torch_available:
            data = torch.from_numpy(data).float().cuda()
            subspaces = torch.from_numpy(np.asarray(self.subspaces)).float().cuda()
        #

        pred_labels_internal = np.zeros(data.shape[0])

        for i in range(data.shape[0]):
            angles = []
            if torch_available:
                for j in range(0, len(self.subspaces)):
                    angles.append(principal_angles_pytorch(data[i,:].view((-1,1)), subspaces[j]))
            else:
                for j in range(0, len(self.subspaces)):
                    angles.append(principal_angles(data[i,:].reshape((-1,1)), self.subspaces[j]))
            pred_labels_internal[i] = self.subspaces_label[angles.index(min(angles))]
        #

        # pred_labels = pred_labels.astype(int)
        # self.results['pred_labels'] = pred_labels

        # pred_labels = self._process_output_labels(pred_labels_internal)
        # return pred_labels
        return pred_labels_internal
    #

    def visualize(self,data):
        labels = self.predict(data)
        import calcom.plot_wrapper as plotter
        plotter.scatter(data[:,0],data[:,1],labels,title="SubspaceMethod")

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

def principal_angles(x, y):
    import numpy as np
    from numpy.linalg import svd, matrix_rank, qr
    angles = []
    qx, _ = qr(x)
    qy, _ = qr(y)
    _, C, _ = svd((qx.transpose()).dot(qy), full_matrices=False)
    rank_x = matrix_rank(x)
    rank_y = matrix_rank(y)
    if rank_x >= rank_y:
        B = qy - qx.dot((qx.transpose()).dot(qy))
    else:
        B = qx - qy.dot((qy.transpose()).dot(qx))
    _, S, _ = svd(B, full_matrices=False)
    S_sorted = np.sort(S)
    for i in range(min([rank_x, rank_y])):
        if C[i]**2 < 0.5:
            angles.append(np.arccos(C[i]))
        elif S_sorted[i]**2 <= 0.5:
            angles.append(np.arcsin(S_sorted[i]))
    return angles

def principal_angles_pytorch(x, y):
    import numpy as np
    from calcom.utils.linalg_pytorch import pytorch_matrix_rank
    global torch
    
    # print small warning if torch version looks old
    if torch.__version__<="1.2.0":
        print('The installed version of torch, %s, appears old. torch.svd() may not be supported. Trying anyway.'%torch.__version__)
    #

    from torch import mm, qr, svd

    # if torch and torch.cuda.is_available() and use_cuda:
    #     x = torch.from_numpy(x).float().cuda()
    #     y = torch.from_numpy(y).float().cuda()
    # elif torch:
    #     x = torch.from_numpy(x).float()
    #     y = torch.from_numpy(y).float()
    angles = []
    qx, _ = qr(x)
    qy, _ = qr(y)
    _, C, _ = svd(qx.t().mm(qy))
    rank_x = pytorch_matrix_rank(x)
    rank_y = pytorch_matrix_rank(y)
    if rank_x >= rank_y:
        B = qy - qx.mm(qx.t()).mm(qy)
    else:
        B = qx - qy.mm(qy.t()).mm(qx)
    _, S, _ = svd(B)
    S_sorted,_ = torch.sort(S)
    for i in range(min([rank_x, rank_y])):
        if C[i]**2 < 0.5:
            angles.append(np.arccos(C[i]))
        elif S_sorted[i]**2 <= 0.5:
            angles.append(np.arcsin(S_sorted[i]))
    return angles
