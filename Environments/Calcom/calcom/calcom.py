# -*- coding: utf-8 -*-

'''Main module.'''
#from __future__ import absolute_import, division, print_function

from .classifiers._abstractclassifier import AbstractClassifier
from .metrics._abstractmetric import AbstractMetric
from .preprocessors import AbstractPreprocessor
from .visualizers._abstractvisualizer import AbstractVisualizer
from .synthdata import AbstractSynthGenerator

# from .utils.smote import smote
# from .utils import smote, smote_multiclass
from .io import CCList

#
# top-level convenience functions
#

def load_style():
    '''
    Loads formatting for our visual "style" by changing matplotlib.rcParams.
    For now, don't force it on-load of importing; user may not wish to
    tweak rcParams if only using a classifier.
    '''
    from calcom.utils import matplotlib_style
    return
#

####################################

class Calcom(object):
    def __init__(self):
        print('WARNING: the Calcom class is severely deprecated and only remains for backward compatibility. Do not use this for new scripts!')
        return
    #

    def load_preprocessor(self, module, path=None):
        '''
        Args:
            - module: file that contains a preprocessor which implements the
              AbstractPreprocessor

        Returns:
            a reference to the first preprocessor class found in the module
        '''
        return load_module_util(module=module,classname=AbstractPreprocessor,classname_str='AbstractPreprocessor',path=path)

    def load_classifier(self, module, path=None):
        '''
        Args:
            - module: file that contains a classfier which implements the
                AbstractClassifier

        Returns:
            a reference to the first classifier class found in the module
        '''
        return load_module_util(module=module,classname=AbstractClassifier,classname_str='AbstractClassifier',path=path)

    def load_visualizer(self, module, path=None):
        '''
        Args:
            - module: file that contains a visualizer which implements the
              AbstractVisualizer

        Returns:
            a reference to the first visualizer class found in the module
        '''
        return load_module_util(module=module,classname=AbstractVisualizer,classname_str='AbstractVisualizer',path=path)

    def load_metric(self,module, path=None):
        '''
        Args:
            - module: file that contains a metric which implements the
              AbstractMetric

        Returns:
            a reference to the first metric class found in the module
        '''
        return load_module_util(module=module,classname=AbstractMetric,classname_str='AbstractMetric',path=path)


def load_module_util(module,classname,classname_str,path=None):
    '''
    Utility function that helps loading a python script dynamically
    Inputs:
            module: file that contains a python script containing the classname
            classname: the class to be loaded
            classname_str: classname in string format
            path: location of the python script in the disk
    Outputs:
            C: the definition of the classname that can be instantiated to create a python object (if the classname exists in the python script)
    '''
    if path:
        import sys
        import os
        sys.path.append(os.path.expanduser(path))
    import importlib
    X = importlib.import_module(module)
    C = None
    for i in dir(X):
        C = getattr(X,i)
        import inspect
        if i != classname_str and inspect.isclass(C) and issubclass(C,classname):
            break
    else:
        raise NotImplementedError("No class in the module "+ module +" extends " + classname_str)
    return C


def main(args=None):
    ccom = Calcom()
    return


if __name__ == "__main__":
    main()
