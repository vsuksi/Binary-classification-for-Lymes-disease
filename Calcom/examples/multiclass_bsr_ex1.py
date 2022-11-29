if __name__ == "__main__":
    '''
    This code is a sanity check on the Balanced Success Rate
    code for a multiclass problem.

    Balanced Success Rate is our term for the average of the
    "true" rates for each class. If there are two classes,
    this is 0.5*( (true positive rate) + (true negative rate) ).
    In general this is 1/n*( sum( true class_i rate ) ).
    '''

    import calcom
    import numpy as np

    bsr = calcom.metrics.ConfusionMatrix(return_measure='bsr')

    labels_preds = []

    labels_true  =       np.array( [0,0,0,0, 1,1,1,1, 2,2,2,2,2,2] )
    labels_preds.append( np.array( [0,0,0,0, 1,1,1,1, 2,2,2,2,2,2] ) ) # Expected bsr: 0.5
    labels_preds.append( np.array( [0,0,1,1, 1,1,0,2, 2,2,2,2,0,1] ) ) # Expected bsr: 0.555...
    labels_preds.append( np.array( [0,0,0,1, 0,1,1,1, 0,0,0,0,0,2] ) ) # Expected bsr: 0.555...
    labels_preds.append( np.array( [0,0,0,0, 0,1,1,1, 0,0,0,2,2,2] ) ) # Expected bsr: 0.75...
    labels_preds.append( np.array( [1,1,1,1, 2,2,2,2, 0,0,0,0,0,0] ) ) # Expected bsr: 0.
    labels_preds.append( np.array( [1,1,1,0, 2,2,2,2, 0,0,0,0,0,0] ) ) # Expected bsr: 0.0833...
    labels_preds.append( np.array( [0,0,0,0, 2,1,1,1, 2,2,2,2,2,2] ) ) # Expected bsr: 0.9166...

    expected_bsrs = [1., (0.5+0.5+4./6)/3., (0.75+0.75+1./6)/3., (1.+0.75+0.5)/3. , 0., (1./4)/3., (1. + 0.75 + 1.)/3.]

    for i,labs in enumerate(labels_preds):

        print("True labels/predicted labels:")
        print(labels_true)
        print(labs)

        print("%15s %.10f" % ("Expected BSR:", expected_bsrs[i]) )
        print("%15s %.10f" % ("Evaluated BSR:", bsr.evaluate(labels_true,labs) ) )
        print("")
    #
