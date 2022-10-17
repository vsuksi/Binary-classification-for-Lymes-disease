from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from six import with_metaclass

class AbstractPreprocessor(with_metaclass(ABCMeta,object)):
    #__metaclass__ = ABCMeta
    
    @abstractmethod
    def process(self):
        pass