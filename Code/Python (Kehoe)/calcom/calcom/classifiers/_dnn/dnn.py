#from __future__ import absolute_import, division, print_function
from calcom.classifiers._abstractclassifier import AbstractClassifier

# from calcom.classifiers.centroidencoder.utilityDBN import standardizeData
# import numpy as np

class DNNClassifier(AbstractClassifier):
    """
        Simple implementation of a DNN using tflearn
    """
    def __init__(self):
        self.params = {}
        self.params['hidden_layer_size'] = [5]
        self.params['hidden_layer_activation'] = ['tanh']

        self.params['output_layer_activation'] = 'softmax'

        self.params['n_epoch'] = 100
        self.params['batch_size'] = 0   # If zero, all data is used.
        self.params['show_metric'] = False

        self.params['auto_layer_structure'] = False

        self.results = {}
        self.results['pred_labels']=[]

        super().__init__()
    #

    @property
    def _is_native_multiclass(self):
        return True
    #
    @property
    def _is_ensemble_method(self):
        return False

    def _fit(self, data, labels):

        import numpy as np
        from calcom.classifiers._centroidencoder.utilityDBN import standardizeData

        # internal_labels = self._process_input_labels(labels)
        internal_labels = labels

        n,d = np.shape(data)
        nclasses = len(self._label_info['unique_labels_mapped'])

        if self.params['auto_layer_structure']:
            d1 = min( int(np.sqrt(d)), d)
            d2 = min( max(3,int(np.sqrt(d1))), d1)  # blah
            self.params['hidden_layer_size'] = [d1,d2]
            self.params['hidden_layer_activation'] = ['tanh','tanh']
        #

        self._mu, self._std, data2 = standardizeData(data)

        p = self.params
        internal_labels = np.reshape(internal_labels, (-1, 1))
        labels2 = np.array(internal_labels==self._label_info['unique_labels_mapped'], dtype=float)

        self.model = init_model(self,d,nclasses)
        # self.model.fit(data,labels2, n_epoch=p['n_epoch'], batch_size=p['batch_size'], show_metric=p['show_metric'], snapshot_epoch=False)
        if self.params['batch_size']:
            self.model.fit(data,labels2, n_epoch=p['n_epoch'], batch_size=self.params['batch_size'], show_metric=p['show_metric'], snapshot_epoch=False)
        else:
            self.model.fit(data,labels2, n_epoch=p['n_epoch'], batch_size=n, show_metric=p['show_metric'], snapshot_epoch=False)
        #
        return
    #

    def _predict(self, data):
        import numpy as np
        from calcom.classifiers._centroidencoder.utilityDBN import standardizeData

        data2 = np.array( standardizeData(data,self._mu,self._std) )

        labeled_data = self.model.predict(data2)
        # labeled_data = np.array( labeled_data, dtype='i' )
        # labeled_data = labeled_data.flatten()

        # Simple majority vote
        pred_labels_internal = np.array([np.argmax(ld) for ld in labeled_data], dtype=int)

        # self.results['pred_labels']=pred_labels
        # pred_labels = self._process_output_labels(pred_labels_internal)
        # return pred_labels
        return pred_labels_internal
    #

    def visualize(self,*args):
        pass

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

def init_model(obj,d,nclasses):
    '''
    inputs:
        obj: the DNNClassifier object (only extracts the parameters)
        d: dimension of a single sample
        nclasses: number of classes in the training.
    '''
    import logging
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tflearn
    import tensorflow as tf

    # Try to get rid of these iteration prints, they're lagging everything.
    tf.logging.set_verbosity(tf.logging.FATAL)


    p = obj.params
    tf.reset_default_graph()
    net  = tflearn.input_data(shape=[None,d])

    for size, act in zip(p['hidden_layer_size'],p['hidden_layer_activation']):
        net = tflearn.fully_connected(net, size, activation=act)

    net = tflearn.fully_connected(net, nclasses, activation=p['output_layer_activation'])
    net = tflearn.regression(net)
    model = tflearn.DNN(net)

    return model
