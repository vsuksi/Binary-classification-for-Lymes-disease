from ._randomclassifier import RandomClassifier
from ._centroidencoder.centroidencoderclassifier import CentroidencoderClassifier
from ._centroidencoder.sparsecentroidencoderclassifier import SparseCentroidencoderClassifier
from ._GrModel import GrModel
from ._ssvm import SSVMClassifier
from ._treeclassifier import TreeClassifier
from ._neuralnetworkclassifier import NeuralnetworkClassifier
from ._ABSClassifier import ABSClassifier

# meta classifiers
from ._multiclass import Multiclass
from ._ensemble.ensembleclassifier import EnsembleClassifier
from ._skeletonensemble import SkeletonEnsemble


# Check for torch -- only import if we can succeed in importing torch **at this level**
try:
    import torch
    have_torch = True
except:
    have_torch = False
#

# torch-dependent classifiers
if have_torch:
    from ._centroidencoder.centroidencodeclassifierPyTorch import CentroidencoderClassifierPyTorch
    from ._neuralnetworkclassifierpytorch import NeuralnetworkClassifierPyTorch
#

# legacy
# from ._abstractclassifier import AbstractClassifier
# from ._nmf.nmfclassifier import NMFClassifier # not a classifier
# from ._dnn.dnn import DNNClassifier # Disabling for the moment

# don't use these
# from ._rfclassifier import RFClassifier

