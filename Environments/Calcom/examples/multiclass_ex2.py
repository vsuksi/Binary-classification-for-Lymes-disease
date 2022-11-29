if __name__ == "__main__":
    '''
    In this example we're looking at tools and extra information stored in
    ConfusionMatrix() and Experiment() classes.

    We demonstrate two things here:
    1. The ability to quickly concatenate the results across
        a cross-validation scheme produced by the Experiment() class,
        using an overloaded sum() operator with the ConfusionMatrix() class;
    2. Tools in the ConfusionMatrix() class to visualize the confusion
        matrix as a sequence of bar plots, and optionally produce a
        stacked bar plot based on a secondary categorical variable.

        For example, if you're classifying the animal, and the secondary
        label is whether or not they're a member of the family Felidae.


    '''
    import numpy as np
    from matplotlib import pyplot

    import calcom

    ############################
    #

    # Generate 4 groups of normally distributed data with some separation.
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

    g0 = normal2d(100,[2,2],[0.5,0.5])
    g1 = normal2d(100,[2,-2],[0.5,0.5])
    g2 = normal2d(100,[-2,2],[0.5,0.5])
    g3 = normal2d(100,[-2,-2],[0.5,0.5])

    data = np.vstack( (g0,g1,g2,g3) )

    # Corresponding labels for the points.
    labels = np.hstack(
        [
            [animal for j in range(100)]
            for animal in ['lion','tiger','bear','bobcat']
        ]
    )

    # Relabel some of the points in advance.
    # We expect the first and last groups being highly misclassified.
    nfake = 50
    labels[-nfake:] = list( np.random.choice(np.unique(labels),nfake) )
    labels[:nfake] = list( np.random.choice(np.unique(labels),nfake) )

    # Scatterplot the data, color by label.
    palette = pyplot.cm.viridis(np.linspace(0,1,4))
    eq = { l:np.where(labels==l)[0] for l in np.unique(labels) }

    fig0,ax0 = pyplot.subplots(1,1)
    for i,(k,v) in enumerate( eq.items() ):
        ax0.scatter(data[v,0],data[v,1], c=palette[i], label=k)

    ax0.legend(loc='upper right')

    # Now we want to
    #   a. Perform the multiclass classification;
    #   b. look at the confusion matrix. We'd like to see
    #       if misclassifications are overrepresented in the groups we've
    #       intentionally mislabeled.

    secondary_labels = ['fake label' if (i>=400-nfake or i<nfake) else 'correct label' for i in range(400)]

    # Set up the multiclass classifier based on SSVM and
    # perform the cross-validation scheme.
    ssvm = calcom.classifiers.SSVMClassifier()
    multi = calcom.classifiers.Multiclass(ssvm)
    cce = calcom.Experiment(
        data,
        labels,
        [multi],
        cross_validation='stratified_k-fold',
        folds=5,
        evaluation_metric=calcom.metrics.ConfusionMatrix('bsr')
    )
    _ = cce.run()

    #############
    #
    # Set up the confusion matrix and evaluate what's happening.
    #

    #
    # Here's where some magic happens.
    # We've overloaded the sum/addition operators with the
    # ConfusionMatrix() class. Essentially what happens is the
    # stored true/predicted labels are concatenated and a new
    # confusion matrix is recomputed. This is useful if you're
    # trying to get summary data across all folds of a
    # cross-validation scheme.
    #
    confmats = cce.classifier_results['Multiclass_0']['confmat']
    confmat = sum(confmats)

    fig,ax = confmat.visualize(
        type='barplot',
        secondary_labels=secondary_labels,
        rows=['lion','tiger','bear','bobcat'], 
        cols=['lion','tiger','bear','bobcat'],
        show=True
    )

    fig.subplots_adjust(left=0.2)

    fig0.show()
    fig.show()
