from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
import os
# import pickle
from six import with_metaclass


class AbstractSynthGenerator(with_metaclass(ABCMeta,object)):
    #__metaclass__ = ABCMeta

    def __init__(self):
        class_name = self.__class__.__name__
        filepath = os.path.expanduser("~/calcom/classifiers/params/" + class_name + ".params")

        if os.path.exists(filepath):
            # with open(filepath,"rb") as f:
            #     self.params = pickle.load(f)
            from calcom.io import load_pkl
            self.params = load_pkl(filepath)


    @abstractmethod
    def fit(self,data,labels):
        '''
        Args:
            - data: training data
            - labels: training labels
        '''
        pass

    @abstractmethod
    def generate(self, synthlabels):
        '''
        Args:
            - synthlabels: labels of desired data.
        '''
        pass

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


    def __str__(self):
        string = ', '.join("{!s}={!r}".format(key,val) for (key,val) in self.params.items())
        return self.__class__.__name__ + '(' + string + ')'

    def __repr__(self):
        return self.__str__()
