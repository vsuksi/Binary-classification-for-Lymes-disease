if __name__ == "__main__":
    '''
    Demonstrating use of our Multiclass() classifier.
    This classifier works by taking an underlying
    (typically binary) classifier and training
    all possible pairwise models between
    pairs of classes. New data is classified by
    taking the majority class vote across all models.

    This script demonstrates this on synthetic data.
    '''
    import numpy as np
    from sklearn import svm,neighbors
    from matplotlib import pyplot

    import calcom

    ############################
    #

    # Generate 3 groups of normally distributed data with some separation.
    def normal2d(n,mu,sigma):
        '''
        inputs:
            n: number of points
            mu: x,y coordinates indicating mean
            sigma: standard deviation in x,y directions (no skew terms allowed!)
        outputs:
            n-by-2 array of datapoints.
        '''
        import numpy as np
        out = np.random.randn(n,2)
        out[:,0] = mu[0] + sigma[0]*out[:,0]
        out[:,1] = mu[1] + sigma[1]*out[:,1]
        return out
    #

    g0 = normal2d(100,[-1,1],[0.5,0.5])
    g1 = normal2d(100,[0,-2],[0.5,0.5])
    g2 = normal2d(100,[1,1],[0.5,0.5])

    data = np.vstack( (g0,g1,g2) )
    labels = np.hstack( [i*np.ones(100) for i in range(3)] )

    # Shuffle the data so that the fit() functions see data
    # from all three classes.
    n,d = np.shape(data)
    shuffle = np.random.permutation(n)

    data = data[shuffle]
    labels = labels[shuffle]

    # Make a list of classifiers to test out.
    clfs = [
        calcom.classifiers.SSVMClassifier(),
        svm.LinearSVC(),
        neighbors.KNeighborsClassifier()
    ]

    # Visualize original data.
    fig,ax = pyplot.subplots(1,len(clfs)+1, figsize=(4*(len(clfs)+1),4), sharex=True, sharey=True)
    ax[0].scatter(data[:,0], data[:,1], c=labels, cmap=pyplot.cm.rainbow)
    ax[0].set_title('Original data colored by label')

    # Run through the classifiers; adapt them into multiclass classifiers,
    # fit and predict on training data, and visualize.
    split = 3*n//4
    for i,clf in zip( np.arange(1,len(clfs)+1) , clfs ):
        # binary_clf = calcom.classifiers.SSVMClassifier()

        multi_clf = calcom.classifiers.Multiclass(clf)

        multi_clf.fit(data[:split], labels[:split])
        pred_lab = multi_clf.predict(data[split:])

        ax[i].scatter(data[:split,0], data[:split,1], c=labels[:split], cmap=pyplot.cm.rainbow)
        ax[i].scatter(data[split:,0], data[split:,1], c=pred_lab, edgecolor='k', linewidth=2,marker='s', cmap=pyplot.cm.rainbow)
        ax[i].scatter([],[],c='w',edgecolor='k',linewidth=2,marker='s',label='Predicted label')

        ax[i].set_title('Data with predicted labels overlaid\n(binary classifier: %s)\n(technique: %s)' % (clf.__class__.__name__,multi_clf.params['method']))
        ax[i].legend(loc='upper right')
    #

    fig.tight_layout()
    fig.show()
