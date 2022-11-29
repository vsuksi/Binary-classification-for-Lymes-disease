
if __name__ == "__main__":
    import calcom
    import numpy as np
    from matplotlib import pyplot


    ##############################
    #
    # Another test using some artificial data and
    # the classifier based on non-negative matrix factorization.
    #
    # If artificial.csv doesn't exist in the same directory, run
    # make_artificial_data.py.
    #
    # The convention here is that the labels are in the last column of the spreadsheet.
    #

    # ccom = calcom.Calcom()
    # data,labels = ccom.load_data('../data/artificial.csv')
    gr_model = calcom.classifiers.GrModel()
    gr_model.init_params({'pair_distance':'euclidean'})
    gr_model.fit(data,labels)
    labels_pred = gr_model.predict(data)

    acc = calcom.metrics.Accuracy()
    myacc = acc.evaluate(labels, labels_pred)

    gr_model.visualize(data)

    print("Accuracy: "+str(myacc))

    ##############################
    #
    # Look at the results.
    #

    fig,ax = pyplot.subplots(1,2, figsize=(10,5))
    ax[0].scatter(data[:,0],data[:,1],c=labels,edgecolor='k',s=20,lw=0,alpha=0.5)
    ax[1].scatter(data[:,0],data[:,1],c=labels_pred,edgecolor='k',s=20,lw=0,alpha=0.5)

    pyplot.show(block=True)
