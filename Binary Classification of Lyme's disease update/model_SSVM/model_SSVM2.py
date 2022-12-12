import calcom
import os
import pandas
import numpy as np

train = pandas.read_csv("train_data.csv")
train_labels = train[".outcome"]
train_features = train.drop(".outcome", axis=1)

test = pandas.read_csv("test_data.csv")
test_features = test.drop(".outcome", axis=1)
test_labels = test[".outcome"]

ssvm = calcom.classifiers.SSVMClassifier()
ssvm.params['C'] = 0.75
ssvm.fit(train_features, train_labels)
pred = ssvm.predict(test_features)

acc = calcom.metrics.ConfusionMatrix('acc')
print(acc.evaluate(pred, test_labels))

#features = pdfeatures.values

#1: 0.9565217391304348
#2:
