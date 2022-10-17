from calcom.synthdata import AbstractSynthGenerator

class DWSmoteGenerator(AbstractSynthGenerator):
    #__metaclass__ = ABCMeta

    def __init__(self):
        self.data = []
        self.labels = []

        self.params = {}
        self.params['eps'] = 0.     # regularization parameter in weighting function
        self.params['power'] = 4    # power for weighting function 1./(d(xi,xj)+eps)**power

        self.results = {}
    #

    def fit(self,data,labels=[]):
        '''
        Args:
            - data: training data
            - labels: training labels. If not specified, assumed to be all of theh same class.
        '''
        import numpy as np
        from scipy.spatial import distance_matrix

        m,n = np.shape(data)

        self.data = np.array(data)
        if len(labels)==0:
            self.labels = np.zeros(m)
        else:
            self.labels = np.array(labels)
        #

        d_mat = distance_matrix(data,data)
        d_mat[d_mat==0.] = np.inf

        labels_u = np.unique(self.labels)

        self.results['eq_tr'] = {lu: np.where(labels==lu)[0] for lu in labels_u}
        # self.results['nearest_neighbors'] = []
        self.results['weights'] = []
        # for lu in labels_u:
        #     self.results['nearest_neighbors']

        # Get the k nearest neighbors *belonging to the same class* to the point.
        # self.results['nearest_neighbors'] = {}
        for i in range(m):
            all_dist = d_mat[i]
            class_neighbors = self.results['eq_tr'][self.labels[i]]
            weights = np.zeros(m)
            # import pdb
            # pdb.set_trace()
            weights[class_neighbors] = np.array((all_dist[class_neighbors] + self.params['eps'])**-self.params['power'])
            weights /= np.sum(weights)
            self.results['weights'].append(weights)

            # sameclass = all_neighbors[ self.results['eq_tr'][self.labels[i]] ]
            # closest_rel = np.argsort(sameclass)[1:self.params['k']+1]
            # closest = self.results['eq_tr'][self.labels[i]][closest_rel]

            # self.results['nearest_neighbors'][i] = closest
        #

        return
    #

    def generate(self, synthlabels):
        '''
        Generate synthetic labels based on the input data.

        Inputs:
            synthlabels : list-like of labels for which corresponding synthetic
                data is requested.
        Outputs:
            synthdata : numpy array of shape len(synthlabels)-by-n, where n is
                the dimensionality of the input data.
        '''
        import numpy as np
        m,n = np.shape(self.data)
        ms = len(synthlabels)

        synthlabels_u = np.unique(synthlabels)
        synthdata = np.zeros( (ms, n) )

        eq_sl = {slu: np.where(synthlabels==slu)[0] for slu in synthlabels_u}

        # nn = self.results['nearest_neighbors']
        eq_tr = self.results['eq_tr']


        for i,slu in enumerate(synthlabels_u):
            p_t = len(eq_tr[slu])   # Number of data points of the class in the training data
            p_s = len(eq_sl[slu])   # Number of requested synth data points in the class

            if p_s%p_t==0:  # If we happened to request a perfect multiple of the original data...
                nreps = p_s//p_t
            else: # Else, round up to the next multiple, and later shuffle the remainder data to avoid possible biases.
                nreps = p_s//p_t + 1
            #

            # Loop over the data points in the training data.
            newData = np.zeros( (p_t*nreps, n) )
            # import pdb
            # pdb.set_trace()
            idx = 0
            for k,ptr in enumerate(eq_tr[slu]):
                cmf = np.cumsum( self.results['weights'][ptr] )
                # selections = np.random.choice(nn[ptr], nreps, replace=True)
                selections = [np.where(np.random.rand() < cmf)[0][0] for i in range(nreps)]

                for j in selections:

                    # diff = _data[neighbor_index,:] - _data[index,:]
                    t = np.random.rand()
                    newData[idx] = (1-t)*self.data[ptr] + t*self.data[j]

                    idx += 1
                #
            #

            # Collect the appropriate data within the class.
            part0 = newData[:(nreps-1)*p_t]
            shuffle = np.random.permutation(np.arange((nreps-1)*p_t,p_s))
            part1 = newData[shuffle]
            synthdata_slu = np.vstack( (part0,part1) )

            # Merge in to master array.
            synthdata[eq_sl[slu]] = synthdata_slu
        #

        return synthdata
    #
#

if __name__ == "__main__":
    from matplotlib import pyplot
    import numpy as np

    smote = DWSmoteGenerator()

    m = 20
    p = 10000
    np.random.seed(57721)   # Euler-Mascheroni

    props = dict(boxstyle='square', facecolor=[0.95,0.95,0.95], edgecolor='k', alpha=0.5)

    th = 2*np.pi*np.random.rand(m)
    x = np.cos(th)
    y = np.sin(th)

    data = np.vstack((x,y)).T
    labels = np.zeros(m)

    fig,ax = pyplot.subplots(1,4, sharex=True, sharey=True, figsize=(14,4))

    # data_synth_all = [[smote_oneclass(data,p,k) for k in [1,2,3,4]]]
    data_synth_all = []
    for k in [1,2,3,4]:

        smote.params['power'] = k
        smote.fit(data,labels)

        data_synth_all.append( smote.generate(np.zeros(p)) )
    #

    for j,k in enumerate([1,2,3,4]):
        ax[j].scatter(data_synth_all[j][:,0], data_synth_all[j][:,1], c='r', s=1)
        ax[j].scatter(data[:,0],data[:,1], c='k', marker=r'$\odot$', s=100)

        ax[j].set_title('power=%i'%k)

        for col,lab,marker,si in [['k','Original data',r'$\odot$',100],['r','Synthetic data','.',20]]:
            ax[j].scatter([],[],c=col,label=lab, marker=marker, s=si)
        #
        ax[j].axis('equal')
    #
    ax[3].legend(loc='upper right', fontsize=10)

    fig.tight_layout()
    fig.suptitle('SMOTE, varying number of nearest neighbors', fontsize=18)
    fig.subplots_adjust(top=0.85)

    ##############################
    #
    # Example 2, with multiple classes.
    #

    m = 20      # Data points per circle
    p = 10000    # Total additional data points.
    np.random.seed(57721)   # Euler-Mascheroni

    props = dict(boxstyle='square', facecolor=[0.95,0.95,0.95], edgecolor='k', alpha=0.5)

    th = 2*np.pi*np.random.rand(m)
    x0 = np.cos(th)
    y0 = np.sin(th)
    th = 2*np.pi*np.random.rand(m)
    x1 = np.cos(th) + 0.5
    y1 = np.sin(th)

    x = np.hstack((x0,x1))
    y = np.hstack((y0,y1))

    data = np.vstack((x,y)).T
    labels = np.hstack( (np.zeros(m), np.ones(m)) )

    fig2,ax2 = pyplot.subplots(1,4, sharex=True, sharey=True, figsize=(14,4))

    for i,k in enumerate([1,2,3,4]):
        synth_labels = np.random.choice([0,1], p)
        colors = ['b' if s else 'r' for s in synth_labels]

        smote.params['power'] = k
        smote.fit(data,labels)

        data_synth = smote.generate(synth_labels)

        ax2[i].scatter(data_synth[:,0], data_synth[:,1], c=colors, s=1)
        ax2[i].scatter(data[:,0],data[:,1], c='k', s=40)
        ax2[i].axis('equal')
        ax2[i].text(0.25,0.9, r'$k=%i$'%k, fontsize=14, transform=ax[i].transAxes, ha='center', va='center', bbox=props)
    #
    fig2.tight_layout()
    fig2.suptitle('SMOTE, varying number of nearest neighbors', fontsize=18)
    fig2.subplots_adjust(top=0.85)


    pyplot.show(block=False)
#


###########################################################
###########################################################
#
# Kept for historical reasons
#
########################################
#
#
# def dw_smote(data,p, **kwargs):
#     '''
#     Alias for calcom.utils.smote.dw_smote_oneclass.
#     '''
#     return dw_smote_oneclass(data,p, **kwargs)
# #
#
# def dw_smote_oneclass(data, p, **kwargs):
#     '''
#     Distance-weighted SMOTE.
#
#     A variation of the original algorithm I'm trying out.
#     Rather than use a uniform
#     random choice of your k nearest neighbors with k as an
#     ad-hoc parameter, instead use a probability function
#     weighted by (inverse) distance to the nearest points.
#
#     By default, an inverse square law of distance is used;
#     that is, the probability that a pair of points (xi,xj)
#     have a synthetic point generated between them is
#     proportional to d(xi,xj)^-2 (rescaled to be a pmf).
#
#     The input data is assumed to be all of the same
#     class.
#
#     In this implementation, an integer number of additional data points can
#     be requested.
#
#     Inputs:
#         data: m-by-n array of the data, with m data points in dimension n.
#             All data is assumed to be of the same class.
#         p: integer, indicating amount of additional data to generate. If p is
#             less than m, the 'center' points are selected uniform randomly.
#             If p>m and p%m!=0, then the integer part of p/m data points are
#             generated using classical SMOTE and the remainder is generated
#             from randomly selected points.
#
#     Optional inputs:
#         eps: float or similar. A regularization term in the non-normalized
#             mass function 1./(d(xi,xj) + eps)**power. (default: 0.5*min(xi,xj) where nonzero)
#         power: integer (or float). Exponent to use in kernel function 1./(d(xi,xj)+eps)**pow.
#             (default: 4)
#     Outputs:
#         data2: (ratio*m)-by-n array of purely synthetic data.
#
#     '''
#     power = kwargs.get('power', 4)
#     # NOTE: default eps is calculated below.
#
#     import numpy as np
#     from scipy.spatial import distance_matrix
#
#     m,n = data.shape
#
#     # Algorithm: Do a simple loop over the data points doing one
#     # extra iteration than necessary. The remainder (if any)
#     # is selected randomly from the end.
#     if p%m==0:
#         nreps = p//m
#     else:
#         nreps = p//m + 1
#     #
#
#
#     # calculate the distance matrix
#     d_mat = distance_matrix(data,data)
#
#     newData = np.zeros((nreps*m, n))
#
#     # Assign infinity to the diagonal elements,
#     # assuming probability of selection is inversely
#     # related to distance. (note: 1./np.inf=0. and np.exp(-np.inf)=0.)
#     #
#     d_mat[d_mat==0.] = np.inf
#
#     eps = kwargs.get('eps', 0.5*d_mat.min())
#
#     idx = 0
#     for i in range(m):
#         # neighbor_indexes = np.argsort(d_mat[i])[1:k+1]
#         # selections = np.random.choice(neighbor_indexes, nreps, replace=True)
#         weights = np.array((d_mat[i] + eps)**-power)
#         weights /= np.sum(weights)
#         cmf = np.cumsum(weights)
#         selections = [np.where(np.random.rand() < cmf)[0][0] for i in range(nreps)]
#         for j in selections:
#             # diff = _data[neighbor_index,:] - _data[index,:]
#             t = np.random.rand()
#             newData[idx] = (1-t)*data[i] + t*data[j]
#
#             idx += 1
#         #
#     #
#
#     part0 = newData[:(nreps-1)*m]
#     shuffle = np.random.permutation(np.arange((nreps-1)*m,p))
#     part1 = newData[shuffle]
#     return np.vstack( (part0,part1) )
# #
#
# def dw_smote_multiclass(data, labels, synth_labels, **kwargs):
#     '''
#     Convenience function which takes in a data matrix
#     and labels and generates additional data according to the
#     input array synth_labels. The k parameter specifies how
#     many nearest neighbors one has to choose from in SMOTE.
#
#     Inputs:
#         data: m-by-n numpy array of the original data
#         labels: list-like size m of labels for the data. Multiple labels allowed.
#         synth_labels: list-like size s of labels for the desired synthetic data.
#             An error is thrown if there are values in synth_labels that aren't
#             in labels.
#         k: integer indicating number of nearest neighbors to use for SMOTE.
#     Outputs:
#         synth_data: s-by-n numpy array of the synthetic data, where
#             each row is a synthetic data point with corresponding label in
#             synth_labels.
#     '''
#     import numpy as np
#
#     m,n = np.shape(data)
#
#     _data = np.array(data)
#     _labels = np.array(labels)
#     _synth_labels = np.array(synth_labels)
#
#     # Sanity check the labels
#     labels_u = np.unique(labels)
#     synth_labels_u = np.unique(synth_labels)
#     if len(np.setdiff1d(synth_labels_u, labels_u)) > 0:
#         raise ValueError("There are requested labels in the synthetic labels that aren't present in the original data.")
#     #
#
#     # Generate equivalence classes for labels seen in synth_labels_u.
#     eq_o = { sl: np.where(sl==_labels)[0] for sl in synth_labels_u }
#     eq_s = { sl: np.where(sl==_synth_labels)[0] for sl in synth_labels_u }
#
#     # Call smote_oneclass for each subset of synth_labels using appropriate data
#     # in the original dataset.
#
#     synth_data = np.zeros( (len(synth_labels),n) )
#     for sl in synth_labels_u:
#         synth_data[ eq_s[sl] ] = dw_smote_oneclass( _data[eq_o[sl]], len(eq_s[sl]), **kwargs )
#     #
#
#     return synth_data
# #
