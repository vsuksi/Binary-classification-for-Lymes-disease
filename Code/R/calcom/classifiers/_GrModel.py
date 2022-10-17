#from __future__ import absolute_import, division, print_function
from calcom.classifiers._abstractclassifier import AbstractClassifier
#import numpy as np

# import scipy as sp
# import pdb



class GrModel(AbstractClassifier):
    def __init__(self):
        self.params = {}
        self.params['pair_distance'] = 'euclidean'

        self.results = {}
        self.results['pred_labels'] = []

        #self.distance = pair_distance
        self.subspaces = []
        self.subspaces_label = []

        super().__init__()

    @property
    def _is_native_multiclass(self):
        return True # at least I think so...
    #
    @property
    def _is_ensemble_method(self):
        return False

    def _fit(self, data, label):
        #internal_labels = self._process_input_labels(label)
        internal_labels = label

        assert data.shape[0] == len(internal_labels), 'given data and label have different length\n'
        self.subspaces, self.subspaces_label = split_data(data, internal_labels, self.params['pair_distance'])
        assert len(self.subspaces) == len(self.subspaces_label), 'label and subspaces have different length\n'

    def _predict(self, data):
        import numpy as np

        pred_labels_internal = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            angles = []
            for j in range(len(self.subspaces)):
                angles.append(principal_angles(np.array([data[i, :]]).transpose(), self.subspaces[j].transpose()))
            pred_labels_internal[i] = self.subspaces_label[angles.index(min(angles))]

        # pred_labels = pred_labels.astype(int)
        # self.results['pred_labels'] = pred_labels

        #pred_labels = self._process_output_labels(pred_labels_internal)

        return pred_labels_internal
    #

    def visualize(self,data):
        labels = self.predict(data)
        import calcom.plot_wrapper as plotter
        plotter.scatter(data[:,0],data[:,1],labels,title="GrModel")

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()


def split_data(data, label, distance):
    import numpy as np
    subspaces = []
    subspaces_label = []
    for i in np.unique(label):
        datai = data[np.where(label == i)[0], :]
        subspaces.extend(group_data(datai, distance))
        import math
        subspaces_label.extend([i for k in range(int(math.ceil(datai.shape[0]*0.5)))])

    return subspaces, subspaces_label


def group_data(data, distance):
    from scipy.spatial.distance import pdist, squareform
    import numpy as np

    grouped_data = []
    tracker = np.array(range(0, data.shape[0]))
    dist = pdist(data, distance)
    d_mat = squareform(dist)
    for k in range(data.shape[0]):
        d_mat[k, k] = float('inf')

    group_num = np.ceil(data.shape[0]*0.5)
    count = 1
    while count <= group_num:
        [x, y] = np.where(d_mat == d_mat.min())
        d_mat[[x[0], y[0]], :] = float('inf')
        d_mat[:, [x[0], y[0]]] = float('inf')
        tracker[x[0]] = -1
        tracker[y[0]] = -1
        grouped_data.append(data[[x[0], y[0]], :])
        if sum(i != -1 for i in tracker) == 1:
            grouped_data.append(data[np.where(tracker != -1)[0], :])
            break
        count += 1
    return grouped_data


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

def cmdscale(D):
    import numpy as np
    # number of points
    n = len(D)
    # centering matrix
    H = np.eye(n) - np.ones((n, n))/n
    # YY^T
    B = -H.dot(D**2).dot(H)/2
    # diagonalize
    evals, evecs = np.linalg.eigh(B)
    # sort eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
    return Y, evals
