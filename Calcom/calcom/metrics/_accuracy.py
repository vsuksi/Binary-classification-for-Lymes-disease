from __future__ import absolute_import, division, print_function
from calcom.metrics._abstractmetric import AbstractMetric

class Accuracy(AbstractMetric):

    def __init__(self):
        '''
        Setup default parameters
        '''
        self.params = {}

    def evaluate(self,actual_labels, pred_labels):
        '''
        actual_labels: actual labels
        pred_labels: predicted labels
        '''
        from sklearn.metrics import accuracy_score
        return accuracy_score(actual_labels,pred_labels)
