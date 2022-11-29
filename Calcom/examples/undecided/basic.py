import calcom
import numpy as np

data = np.random.randn(100,20)
labels = np.random.choice(['time05','time12','time21','time29'],100)
bsr = calcom.metrics.ConfusionMatrix('bsr')

nnc = calcom.classifiers.NeuralnetworkClassifier()

cce = calcom.Experiment(data,labels,[nnc],cross_validation='stratified_k-fold', folds=3, evaluation_metric=bsr)

result = cce.run()

cf = sum( cce.classifier_results['NeuralnetworkClassifier_0']['confmat'] )
cf.visualize(type='barplot', show=True)
