from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from six import with_metaclass

class AbstractVisualizer(with_metaclass(ABCMeta,object)):
    #__metaclass__ = ABCMeta
    
    @abstractmethod
    def project(self,data):
        '''
        Input: n by m array "data"
        Output: n by d array "coords", reduced to d dimensions, 
        in some sense.
        '''
        pass

    @abstractmethod
    def visualize(self,coords):
        '''
        Input: n by d array "coords", reduced to d dimensions in some sense, 
        '''
        pass
