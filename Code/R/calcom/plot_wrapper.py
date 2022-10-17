from __future__ import absolute_import, division, print_function
'''
This is an alternate interface for matplotlib to display plots with one line of
python.

As of 27 Nov 2019, gutting a lot of the more "basic" functionality;
this needs to be rebuilt and reconsidered.
'''

#import matplotlib
#import numpy as np
#import matplotlib.pyplot as plt

def confusion(data, labels, annotate=False,*args,**kwargs):
    '''
    Confusion matrix plot

    Args:
        - data: NxN matrix
        - labels: size N list of corresponding labels to the data (labels are
          plotted from left to right and up to down)
    '''
    data = np.array(data)

    im = plt.imshow(data, cmap='Reds', interpolation='nearest',*args,**kwargs)
    plt.colorbar(im, orientation='vertical')
    #plt.grid(True)
    #plt.rc('grid', linestyle="-", color='black')

    # setting labels
    index = np.arange(len(labels))
    plt.xticks(index, labels)
    plt.yticks(index, labels)

    # hide ticks
    ax = plt.axes()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # annotations
    if annotate:
        width, height = data.shape
        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(data[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

    plt.show()
#

def scatter_train_test(train_data, train_labels, test_data, test_labels, show=True, *args, **kwargs):
    '''
    Takes in separate test and train data, generates one scatter plot,
    with the purpose of associating training and testing data easily via "paired"
    colormaps (such as pyplot.cm.tab20).

    Named and keyword arguments are passed on to the call to pyplot.plot().
    Otherwise, calcom modifications to matplotlib.rcParams
    (via calcom.utils.matplotlib_style) are used.

    Returns:
        pyplot figure/axis pair.
    '''
    import pdb
#    pdb.set_trace()

    import calcom
    from matplotlib import pyplot
    import numpy as np

    # Input validation
    nR,dR = train_data.shape
    nE,dE = test_data.shape

    if dR!=dE: raise Exception('Training and testing data have different dimensionality; %i!=%i'%(dR,dE) )
    if dR>3: raise Exception('Training data has dimension %i; cannot visualize beyond d=3.'%dR)
    if dE>3: raise Exception('Testing data has dimension %i; cannot visualize beyond d=3.'%dR)

    ####
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    ec_R,ec_E = {}, {}
    ec_R = {j:np.where(train_labels==j)[0] if j not in ec_R else _ for j in train_labels}
    ec_E = {j:np.where(test_labels==j)[0] if j not in ec_E else _ for j in test_labels}

    all_classes = np.unique(np.concatenate( [list(ec_R.keys()), list(ec_E.keys())] ))
    n_classes = len(all_classes)
    # last check
    if n_classes>10: raise Exception('%i classes identified. Only 10 classes are supported.'%n_classes)

    ###################################
    calcom.load_style()

    # need to identify matched pairs of labels and associate color pairs with them.
    full_palette = np.array([pyplot.cm.tab20(j) for j in range(20)])
    tr_color_map = {}
    te_color_map = {}

    for j,classname in zip(range(0,20,2), all_classes):
        tr_color_map[classname] = full_palette[j+1]
        te_color_map[classname] = full_palette[j]

    if dR==3:
        from mpl_toolkits.mplot3d import Axes3D
#        fig,ax = pyplot.subplots(1,1, projection='3d')
        fig = pyplot.figure()
        ax = fig.add_subplot(111,projection='3d')

        for j,ac in enumerate(all_classes):
            if ac in ec_R.keys():
                v = ec_R[ac]
                ax.scatter(train_data[v,0], train_data[v,1], train_data[v,2], lw=0, c=[tr_color_map[ac]], label=ac+' (train)', *args, **kwargs)
            if ac in ec_E.keys():
                v = ec_E[ac]
                ax.scatter(test_data[v,0], test_data[v,1], test_data[v,2], lw=0, c=[te_color_map[ac]], label=ac+' (test)', *args, **kwargs)
    else:
        fig,ax = pyplot.subplots(1,1)

        for j,ac in enumerate(all_classes):
            if ac in ec_R.keys():
                v = ec_R[ac]
                ax.scatter(train_data[v,0], train_data[v,1], lw=0, c=[tr_color_map[ac]], label=ac+' (train)', *args, **kwargs)
            if ac in ec_E.keys():
                v = ec_E[ac]
                ax.scatter(test_data[v,0], test_data[v,1], lw=0, c=[te_color_map[ac]], label=ac+' (test)', *args, **kwargs)
    #

    ax.legend()
    calcom.utils.matplotlib_style.clean_scatter(fig,ax)

    if show: fig.show()
    return fig,ax
#


def scatterExperiment(eObject, data, labels, readable_label_map={}, *args,**kwargs):
    '''
    Takes an Experiment object, creates a multi-axes figure which scatters and
    classifies on a given dataset and labels, all the data with a given
    visualizer (PCA for now), and indicates true labels, predicted labels, etc.

    Args:
        - readable_label_map:
            - Dictonary which maps the values in the labels array to
              human-readable labels.
            - For example: readable_label_map = {0:"Control", 1:"Shedder"}
        .. note::
            Optional arguments are passed to all the scatter commands.
    '''

    from matplotlib import gridspec

    nclass = len(eObject.classifier_list)
    # true_labels = eObject.labels
    true_labels = labels
    unique_true_labels = np.unique(true_labels)
    unique_true_labels.sort()
    nlabels = len(unique_true_labels)

    if (not len(readable_label_map)):
        # Generic labels if human-readable labels aren't specified.
        readable_label_map = {}
        for i,label in enumerate(unique_true_labels):
            readable_label_map[i] = label
        #
    #

    # Generate a mapping from given labels to range(nlabels) for referencing
    # default colors/shapes.
    int_label_map = {}
    for i,lab in enumerate(unique_true_labels):
        int_label_map[lab] = i
    #

    # mapped_true_labels = [ label_map[lab] for lab in labels ]

    # Generate arrays of true colors and shapes for all the points.
    # Predicted colors are generated on a per-classifier basis in the loop.
    truecolors = []
    for i,lab in enumerate(true_labels):
        truecolors.append(PLT_COLOURS[int_label_map[lab]])
    #

    # Set up the figure and axes.
    if (nclass<=4):
        #       nclass <= 3 -> one row
        # fig = plt.figure()
        # ax = [ plt.subplot2grid( (1,nclass), (0,i) ) for i in range(nclass) ]
        fig,ax = plt.subplots(1,nclass, sharex=True, sharey=True, figsize=(5*nclass,5))

        # Manually expand out the bounds of the subplots.
        fig.subplots_adjust(wspace=0.05,left=0.1,right=0.95,top=0.90,bottom=0.15)

    elif (nclass<=8):
        #       nclass <=8  -> two rows
        # print("Visualizing experiments with more than three classifiers is not currently supported.")
        # return
        fig,axtemp = plt.subplots(2,4,sharex=True,sharey=True,figsize=(5*4,10))
        ax = axtemp.flatten()

        # Disable axes not being used.
        for j in range(nclass,8):
            ax[j].axis('off')
        #
        # TODO: Center-align the axes in the bottom half.

    else:
        print("Visualizing experiments with more than eight classifiers is not currently supported.")
        return
    #

    # Project data to 2d only for now (default for PCAVisualizer).
    import calcom
    pcavis = calcom.visualizers.PCAVisualizer()
    projdata = pcavis.project(data,true_labels,[readable_label_map[lab] for lab in unique_true_labels])
    # projdata = eObject.data[:,:2]

    classkeys = list( eObject.best_classifiers.keys() )
    # For each classifier,
    for i,key in enumerate(classkeys):
        # print(key)

        # Create the label mapping for the predicted data; this is done on the fly.
        # pred_labels = eObject.best_classifiers[key].results['pred_labels']
        best_clf = eObject.best_classifiers[key]
        if isinstance(best_clf, list):
            best_clf = best_clf[0]
        pred_labels = best_clf.predict(data)

        predcolors = []
        for lab in pred_labels:
            predcolors.append(PLT_COLOURS[int_label_map[lab]])
        #

        # Map true/predicted classification pairs to integers, then build
        # the four subsets of the apocalypse.
        classtype = np.array( [ 2*int_label_map[true_labels[j]] + int_label_map[pred_labels[j]] for j in range(len(pred_labels)) ] )
        subsets = [ np.where(classtype==j)[0] for j in range(4) ]

        ax[i].scatter(projdata[subsets[0],0], projdata[subsets[0],1], c=PLT_COLOURS[0], marker='o', s=40,  lw=1.5, label=readable_label_map[0], *args, **kwargs )
        ax[i].scatter(projdata[subsets[1],0], projdata[subsets[1],1], c=darken_color(PLT_COLOURS[0]), marker='*', s=40,  lw=1.5, label="False Positive", *args, **kwargs )
        ax[i].scatter(projdata[subsets[2],0], projdata[subsets[2],1], c=darken_color(PLT_COLOURS[1]), marker='*', s=40,  lw=1.5, label="False Negative", *args, **kwargs )
        ax[i].scatter(projdata[subsets[3],0], projdata[subsets[3],1], c=PLT_COLOURS[1], marker='o', s=40,  lw=1.5, label=readable_label_map[1], *args, **kwargs )

        ax[i].set_title(str(key))
        ax[i].legend(loc='upper left')
    #

    return fig

#

if __name__ == "__main__":
    time = [0, 1, 2]
    mobilex = [0, 1, 2]
    mobiley = [0, 1, 2]

    mydict = {"A": 20, "B": 35, "C": 30, "D": 35, "E": 27}
    bar(mydict)


    z=np.array(((21,1,0),
                (0,25,6),
                (0,0,22)))

    labels = ["a", "b", "c"]
    confusion(z, labels)
