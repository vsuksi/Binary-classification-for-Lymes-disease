if __name__ == "__main__":
    '''
    This example shows how you can make a "stacked barplot" visualizing
    a confusion matrix.
    '''

    import numpy as np
    from matplotlib import pyplot
    from calcom.metrics import ConfusionMatrix

    # Create a fake dataset with true/predicted labels,
    # and secondary labels for the stacked barplot.
    choices = ['lion', 'tiger', 'bear']
    secondary_choices = ['petted', 'hugged', 'cuddled']

    t_l = np.random.choice(choices,300) # true labels
    p_l = np.random.choice(choices,300) # predicted

    # Secondary categorical variable.
    secondary_l = np.random.choice(secondary_choices,300)

    # Evaluate the confusion matrix and generate a plot
    # incorporating the secondary labels. Rows indicate
    # all the data corresponding to a true label;
    # each bar is the corresponding count of predicted labels.
    cf = ConfusionMatrix()
    cm = cf.evaluate(t_l, p_l)

    fig,ax = cf.visualize(type='barplot', secondary_labels=secondary_l)
    fig.show()
