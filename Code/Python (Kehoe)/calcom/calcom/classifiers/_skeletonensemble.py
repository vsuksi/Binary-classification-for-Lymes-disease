#from __future__ import absolute_import, division, print_function
from calcom.classifiers._abstractclassifier import AbstractClassifier
from calcom.classifiers._randomclassifier import RandomClassifier


class SkeletonEnsemble(AbstractClassifier):
    '''
    Purpose: This is a generic interface to convert a
    collection of classifiers into an ensemble classifier.

    This differs from Tomojit's Ensemble Classifier
    in that the main purpose is to investigate schemes
    for binary classification where the classification
    is no longer done by majority vote; rather
    it is sensitive to the number of votes cast
    (e.g., if even 1 of 10 classifiers votes for a
    class, then that class is assigned).

    This also differs in that the user must specify
    a list of classifiers in advance. However, this
    may mean that classifiers across multiple
    data types may be used.

    In some sense this is nothing more than application of
    an ROC curve to the ensemble, with the parameter
    being the percent vote for one class.
    '''
    def __init__(self,clf_list=[RandomClassifier()]):
        import copy
        self.params = {}
        self.params['classifiers'] = [copy.deepcopy(clf) for clf in clf_list]
        self.params['vote_threshold'] = 0.5    # 0.5 = simple majority vote
        self.params['verbosity'] = 0

        # What to use for class0? Defaults to the smaller in the ordering
        # of the two labels. vote_ratio < params['vote_ratio'] => class0
        self.params['class0'] = None

        self.params['feature_sets'] = None  # should be a list of pointers, same length as clf_list

        self.results = {}
        pass
    #

    @property
    def _is_native_multiclass(self):
        return False
    #
    @property
    def _is_ensemble_method(self):
        return True
    #

    def _fit(self,data,labels,**kwargs):
        '''
        SkeletonEnsemble fit function. Given the input data and
        labels, build a collection of models from the underlying classifier.

        MUST BE A BINARY CLASSIFICATION. An error will be thrown if
        len(np.unique(labels))>2.

        Inputs:
            data : numpy array size n-by-d with n datapoints in d dimensions
            labels : array or list of size n with corresponding labels.

        Optional inputs:
            feature_sets : A list of pointers indicating subsets of
                dimensions for each classifier to use.

        Outputs:
            None.
        '''
        import numpy as np

        if len(np.unique(labels))>2:
            print(np.unique(labels))
            raise AttributeError("The training data has more than two labels. SkeletonEnsemble only supports binary classification.")
        #

        import copy
        from calcom.io import CCList

        verb = self.params['verbosity']
        n,d = np.shape(data)

        if type(self.params['feature_sets']) == type(None):
            self._fsets = [list(range(d)) for _ in self.params['classifiers']]
        else:
            self._fsets = self.params['feature_sets']
        #

        for i,clf in enumerate(self.params['classifiers']):
            ptrs = self._fsets[i]
            clf.fit(data[:,ptrs],labels,**kwargs)
        #

        return
    #

    def _predict(self,data):

        import numpy as np
        from calcom.io import CCList
        import copy

        # import pdb
        # pdb.set_trace()

        ul2 = self._label_info['unique_labels_mapped']

        nl = len(ul2)
        n,d = np.shape(data)

        all_predictions = []
        for i,dp in enumerate(data):
            # reshape and 0th entry for sklearn compatibility.
            preds = [clf.predict( dp[self._fsets[j]].reshape(1,-1) )[0] for j,clf in enumerate(self.params['classifiers'])]
            all_predictions.append(preds)
        #

        self.results['all_predictions'] = np.array(all_predictions)

        # Conversion from predict to _predict() a bit weird here;
        # going to just replace ul with the mapped ones.
        # ul = copy.copy(self._label_info['unique_labels'])
        ul = copy.copy(self._label_info['unique_labels_mapped'])

        ul_ordered = []
        if type(self.params['class0']) == type(None):
            ul_ordered = ul
            ul_ordered.sort()
        else:
            ul_ordered.append( self.params['class0'] )
            ul.remove( ul_ordered[0] )
            ul_ordered += ul
        #

        # finally time for the tally.
        # vote_ratio < params['vote_ratio'] => class0.
        all_counts = []
        for ap in all_predictions:
            counts = [ap.count(ulo) for ulo in ul_ordered]
            all_counts.append( counts )
        #
        ratios = np.array([c[0] for c in all_counts],dtype=float)/len(ul_ordered)
        pred_labels = np.array(ul_ordered)[ np.array(ratios < self.params['vote_threshold'], dtype=int) ]

        self.results['vote_counts'] = np.array(all_counts)

        return pred_labels
    #

    def visualize(self):
        pass
    #

#
