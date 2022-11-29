if __name__ == "__main__":
    import calcom
    import numpy as np

    # Grab the dataset. The "dataset" submodule loads 
    # a collection of small data sets and puts them in 
    # the CCDataSet format.
    from calcom import datasets
    iris = datasets.iris

    # shuffle the data and look at a visualization.
    order = np.random.permutation( np.arange(len(iris.data)) )

    data,labels = iris.generate_classification_problem('flower_type', idx=order)

    n,d = np.shape( data )

    pca = calcom.visualizers.PCAVisualizer()

    # Build a projector based on a small set of the data and see how it 
    # behaves on the rest of the data. 
    # NOTE: syntax may be cleaned/conform to sklearn in the future.
    split = n//2
    co_train = pca.project( data[:split] )
    labels_train = labels[:split]

    co_test = np.dot( ( data[split:] - pca.results['data_mean'] ), pca.results['components'] )
    labels_test = labels[split:]

    fig,ax = calcom.plot_wrapper.scatter_train_test( co_train, labels_train, co_test, labels_test )

    # note - the iris dataset is well-behaved; very little data is needed to 
    # create a 'meaningful' representation.

    ax.set_xlabel('principal component 1')
    ax.set_ylabel('principal component 2')

    # 
    calcom.utils.matplotlib_style.clean_scatter(fig,ax)

