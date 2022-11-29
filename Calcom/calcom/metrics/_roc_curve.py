#from __future__ import absolute_import, division, print_function
from calcom.metrics._abstractmetric import AbstractMetric
#from sklearn.metrics import accuracy_score

class ROCcurve(AbstractMetric):

    def __init__(self):
        '''
        Setup default parameters

        Parameters:
            ''
        '''
        self.params = {}
        self.params['calculate_auc'] = True
        self.results = {}
    #

    def evaluate(self, labels_true, labels_predicted):
        '''
        ROC curve for binary classification problem
        based on np.unique(labels_true). Generates an ROC curve
        using true labels and real-valued predicted labels.

        The parameterization is done by changing the threshold
        on labels_predicted:

            labels_predicted[labels_predicted<=thresh] = min(labels_true)
            labels_predicted[labels_predicted>thresh] = max(labels_true)

        where thresh varies from min(labels_predicted) to max(labels_predicted).
        Note that this means that if labels_predicted correspond to probabilities,
        you must have it set up so that labels_predicted[i]==1 corresponds to
        labels_predicted[i] mapping to max(labels_true) with probability 1.

        Parameters:
            self.params['calculate_auc']
            If True, calculates the AUC for the resulting ROC curve and stores
            it in self.results['auc'] (default: True)

        Inputs:
            labels_true: 1d numpy array, of the true labels. Can be float or integer.
            labels_predicted: 1d numpy array, of the predicted labels, probabilities, etc.

        Outputs:
            tprs: 1d numpy array of true positive rates as the threshold parameter is varied.
            fprs: 1d numpy array of false positive rates as the threshold parameter is varied.

            These are also saved in self.results['tprs'] and self.results['fprs'].

        '''

        import numpy as np
        from calcom.metrics import ConfusionMatrix

        cf = ConfusionMatrix()

        labels_unique = np.sort(np.unique(labels_true))
        if len(np.unique(labels_true))>2:
            #print("Error: multiclass ROC is not supported.")
            #return
            raise ValueError("Error: multiclass ROC is not supported.")
        #
        if len(labels_true)!=len(labels_predicted):
            #print(len(labels_true))
            #print(len(labels_predicted))
            #print("Error: length of the true and predicted labels must be the same.")
            #return
            raise ValueError("Error: length of the true({}) and predicted labels({}) must be the same.".format(len(labels_true), len(labels_predicted)))
        #

        nlab = len(labels_true)


        threshes = np.array(labels_predicted)
        threshes.sort()
        threshes = np.concatenate( (threshes,[threshes[-1]+1]) )

        fprs = np.zeros(len(threshes))
        tprs = np.zeros(len(threshes))

        for i,thresh in enumerate(threshes):

            labels_mapped = labels_unique[ (labels_predicted >= thresh).astype(int) ]

            cf.evaluate(labels_true, labels_mapped)

            fprs[i] = cf.results['fpr']
            tprs[i] = cf.results['tpr']
        #

        self.results['fprs'] = fprs
        self.results['tprs'] = tprs

        if self.params['calculate_auc']:
            self.results['auc'] = np.trapz(tprs,fprs)
        #

        return fprs,tprs
    #
#
