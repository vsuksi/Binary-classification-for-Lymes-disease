#from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod,abstractproperty
# from abc import abstractmethod,abstractproperty

from six import with_metaclass


###########################

class AbstractClassifier(with_metaclass(ABCMeta,object)):
    #__metaclass__ = ABCMeta

    def __init__(self):
        import os
        class_name = self.__class__.__name__

        # TODO: GENERALIZE THIS; TARGET THE INSTALLATION DIRECTORY, OR
        # SOME OTHER DEFAULT DIRECTORY SPECIFIED ON INSTALLATION OF PACKAGE.
        filepath = os.path.expanduser("~/calcom/classifiers/params/" + class_name + ".params")

        if os.path.exists(filepath):
            # with open(filepath,"rb") as f:
            #     self.params = pickle.load(f)
            from calcom.io import load_pkl
            self.params = load_pkl(filepath)
        #
    #

    #######################
    #
    # These functions are automatically defined for every
    # classifier (that is a subclass of the AbstractClassifier)
    #

    def init_params(self,params):
        '''
        Args:
            params: set one or more parameters as directory
        '''
        if hasattr(self,'params'):
            for key, value in params.items():
                self.params[key] = value
        else:
            raise Exception("params was not set for the classifier")
    #


    def _process_input_labels(self,labels):
        '''
        Takes in external labels and produces a
        dictionary of mapping of the classes to
        internal labels, and vice versa.
        '''
        from calcom.io import CCList
        self._label_info = {}

        ldict = {}
        ildict = {}
        i = 0
        u_l = []
        u_l_mapped = []
        for l in labels:
            if l not in ldict.keys():
                ldict[l] = i
                ildict[i] = l
                u_l.append(l)
                u_l_mapped.append(i)
                i += 1
        #
        self._label_info['ldict'] = ldict
        self._label_info['ildict'] = ildict
        self._label_info['unique_labels'] = u_l
        self._label_info['unique_labels_mapped'] = u_l_mapped

        return CCList([ldict[l] for l in labels])
    #
    def _process_output_labels(self,internal_labels):
        '''
        Takes in internally produced labels and returns
        a list of the labels mapped to the original
        label names.
        '''
        from calcom.io import CCList

        pred_labels_output = CCList( [self._label_info['ildict'][l] for l in internal_labels] )
        if hasattr(self,'results'):
            self.results['pred_labels'] = pred_labels_output
        else:
            self.results = {'pred_labels': pred_labels_output}
        #
        return pred_labels_output
    #

    ##############
    #
    # The methods below must be implemented by the user
    # coding the classifier.
    #


    # Enforce that the implementation of the classifier has a
    # "_is_native_multiclass" method. Use @property in the classifier
    # itself so we can simply type clf._is_native_multiclass and
    # have it be "immutable".
    @abstractproperty
    def _is_native_multiclass(self):
        '''
        Must return True or False
        '''
        pass
    #

    # Enforce that the implementation of the classifier has a
    # "_is_ensemble_method" method. Use @property in the classifier
    # itself so we can simply type clf._is_ensemble_method and
    # have it be "immutable".
    @abstractproperty
    def _is_ensemble_method(self):
        '''
        Must return True or False. 
        NOTE: "ensemble" here refers to the implementation 
            require one or more classifiers on instantiation
            of the class. Hence, a random forest implementation 
            while technically being an ensemble, typically 
            is a collection of classification trees built 
            univariately.
        '''
        pass
    #

#    @abstractmethod
    def fit(self,data,labels):
        '''
        This is the _OUTWARD FACING_ fit function,
        which performs mandatory processing and sanity
        checks on the data and labels.

        Your code should go in _fit(), assuming the
        labels have been mapped to integer values and

        Inputs:
            data : n-by-d array of numerical values
            labels : list-like of n values of labels associated
                with each row of the data.
        '''
        import numpy as np
        # Verify there are no NaN values in the data.
        if np.any(data!=data):
            raise ValueError('There are NaN values in the data array; cannot continue with fitting.')
        #
        # process the labels.
        labels_internal = self._process_input_labels(labels)

        # Make a call to the user's _fit function.
        self._fit(data,labels_internal)
        return
    #

#    @abstractmethod
    def predict(self,data):
        '''
        This is the _OUTWARD FACING_ predict function,
        which performs mandatory sanity checks on the input data
        and post-processes the labels, mapping them back from
        internal to external labels.

        Your code should go in _predict(); that code should
        assume the data has been cleaned of NaN values (all other bets are off).
        '''
        import numpy as np
        if np.any(data!=data):
            raise ValueError('There are NaN values in the data array; cannot continue with classification.')
        #
        pred_labels_internal = self._predict(data)
        pred_labels = self._process_output_labels(pred_labels_internal)

        return pred_labels
    #

    @abstractmethod
    def _fit(self,data,labels):
        '''
        This is the _INTERNAL_ fit function. This is what
        should be defined by the user.

        Inputs:
            data: numpy array of shape (n,d) with numerical values.
            labels: numpy array of shape (n,) with integer values associated with each label.
        '''
        pass

    @abstractmethod
    def _predict(self,data):
        '''
        This is the _INTERNAL_ predict function. This is what
        should be defined by the user. If the algorithm
        maps the classes to something other than a subset of the set
        {0,1,...}, then the user MUST invert the mapping done
        as part of the classifier (I'm looking at you, SVM).

        However, mapping back to the original labels is done automatically
        by self.predict() (no leading underscore).

        Inputs:
            data: numpy array of shape (n,d) with numerical values.
        '''
        pass


    @abstractmethod
    def visualize(self):
        '''
        Adhoc visualizer for the classifier
        '''
        pass

    def __str__(self):
        string = ', '.join("{!s}={!r}".format(key,val) for (key,val) in self.params.items())
        return self.__class__.__name__ + '(' + string + ')'

    def __repr__(self):
        return self.__str__()
