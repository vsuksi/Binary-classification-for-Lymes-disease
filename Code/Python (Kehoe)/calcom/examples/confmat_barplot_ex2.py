if __name__ == "__main__":
    '''
    Demonstration of some of the various options
    available for making a stacked barplot visualization
    of a confusion matrix using randomly generated
    labels.

    Note that our plotting functions are not perfect
    in intelligently managing plot labels, as can be observed
    in some of these plots. You may need
    to manually adjust whitespace or the figure size
    for your use case.
    '''

    import numpy as np
    from matplotlib import pyplot

    # Should provide similar functionality to the prototype, but
    # now we can view the entire confusion matrix.
    #
    # Also, better control of colors.

    # Create a fake dataset with true/predicted labels,
    # and secondary labels for the stacked barplot.
    choices = ['dogs','cats','penguins','lizards','parakeets','humans']
    secondary_choices = ['petted', 'cuddled', 'comforted', 'consoled', 'reassured']
    p_l = np.random.choice(choices,4000)
    t_l = np.random.choice(choices,4000)
    secondary_l = np.random.choice(secondary_choices,4000)

    palette = pyplot.cm.rainbow(np.linspace(0,1,len(choices)))
    secondary_colors = {ch:col for ch,col in zip(secondary_choices,palette)}

    #############################

    from calcom.metrics import ConfusionMatrix

    cf = ConfusionMatrix()
    cm = cf.evaluate(t_l, p_l)

    # With no arguments - a plain barplot.
    fig,ax = cf.visualize(type='barplot',bar_colors='r',show=True)

    # With some arguments - coloring by secondary label; choice of colors.
    # This currently isn't great for a multiclass problem.
    fig2,ax2 = cf.visualize(type='barplot',secondary_labels=secondary_l, bar_colors=secondary_colors, show=True)

    # But not the worst if we only look at two of the classes.
    fig3,ax3 = cf.visualize(type='barplot',
                            secondary_labels=secondary_l,
                            bar_colors=secondary_colors,
                            which=['dogs','cats'],
                            show=True
                        )

    # What about a single input, without colors specified?
    fig4,ax4 = cf.visualize(type='barplot',
                            secondary_labels=secondary_l,
                            # bar_colors=secondary_colors,
                            which='lizards',
                            show=True
                        )
