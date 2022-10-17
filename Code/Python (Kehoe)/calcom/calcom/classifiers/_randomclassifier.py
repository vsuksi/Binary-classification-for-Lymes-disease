#from __future__ import absolute_import, division, print_function
from calcom.classifiers._abstractclassifier import AbstractClassifier

class RandomClassifier(AbstractClassifier):

    def __init__(self):
        '''
        Setup default parameters
        '''
        self.params = {}

        self.results = {}
        self.results['pred_labels'] = []

        super().__init__()
    #

    @property
    def _is_native_multiclass(self):
        return False
    #
    @property
    def _is_ensemble_method(self):
        return False

    def _fit(self,data,labels):
        '''
        data: training data
        labels: training labels
        '''
        # _ = self._process_input_labels(labels)
        return
    #

    def _predict(self,data):
        '''
        data: test data
        '''
        import random
        import numpy as np

        pred_labels_internal = np.random.choice(self._label_info['unique_labels_mapped'], np.shape(data)[0])
        # pred_labels = self._process_output_labels(pred_labels_internal)
        #
        # return pred_labels
        return pred_labels_internal
    #

    def visualize(self,*args):
        pass

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
