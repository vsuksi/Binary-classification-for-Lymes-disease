from __future__ import absolute_import, division, print_function
from calcom.preprocessors import AbstractPreprocessor


class Normalizer(AbstractPreprocessor):
    
    def __init__(self):
        '''
        Setup default parameters
        '''
        self.params = {}
    
    def process(self,data):
        '''
        data: n-by-r array of data, with n the number of observations. 
        '''
            
        return data/data.max(axis=0)