if __name__ == "__main__":
    '''
    This example showcases a basic classification task:
    binary classification on randomly generated data.

    Demonstrates the Sparse SVM (SSVM) classifier
    and our Confusion Matrix class.
    '''
    import calcom
    import numpy as np

    # Randomly generated data and labels
    n,d = 100,5
    data = np.random.randn(n,d)
    labels = np.random.choice(['lion','tiger'], n)

    #
    # Fit on the first half of the data,
    # then predict on the second half.
    #
    # Note that the functions follow the same
    # ".fit()", ".predict()" naming convention as
    # scikit-learn. However, parameters are modified
    # using a dictionary scheme.
    #
    ssvm = calcom.classifiers.SSVMClassifier()
    ssvm.fit(data[:n//2], labels[:n//2])
    pred = ssvm.predict(data[n//2:])

    #
    # Evaluate performance on the test data.
    #
    acc = calcom.metrics.ConfusionMatrix('acc')
    print( acc.evaluate( pred, labels[n//2:] ) )
