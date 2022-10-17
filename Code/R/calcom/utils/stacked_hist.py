def stacked_hist(arr, **kwargs):
    '''
    Creates a histogram of the given array of numerical values.
    Extra functionality to create stacked barplots
    based on an optional input "label".

    Constructed using repeated calls to matplotlib.pyplot.bar()
    with the optional input "bottom" defined, with extra
    bells and whistles added.

    Inputs: arr
    Optional inputs:
        bins: list-like of left/right endpoints for each bin.

        labels : list-like of the same size as arr.
            If specified, a
            stacked barplot is generated which visualizes
            how quantities of a secondary label are assigned in the
            corresponding bins. The entries may be of any type, and
            are labeled in the figure. Default: None

        bar_colors : Either a single tuple, to use a single color for all bars,
            or a dictionary mapping secondary label
            names to colors; internally passed to pyplot.bar.
            If dictionary entries do not exist for required labels,
            colors from pyplot.cm.Set3 are sampled.

        show : Boolean; whether to show the plot immediately. Default: False

    Outputs: pyplot figure, axis pair.
    '''
    from matplotlib import pyplot
    from calcom.utils import type_functions as tf
    import numpy as np

    # Process known data; generate defaults if needed.
    bins = kwargs.get( 'bins', np.linspace(np.nanmin(arr),np.nanmax(arr),11) )

    labels = kwargs.get('labels', np.array([0 for _ in arr]) )
    u_s_l = np.unique(labels)

    if len(u_s_l)==1:
        # Flag to remind ourselves not to later create a legend
        # when one wasn't desired in the first place.
        skip_legend = True
    else:
        skip_legend = False
    #

    secondary_colors = kwargs.get('bar_colors',
        pyplot.cm.Set3( np.mod(3+np.arange(len(u_s_l)),12)/12. )    # Rotated colors of Set3
    )

    if not type(secondary_colors)==dict:
        secondary_colors = {usl: secondary_colors[i] for i,usl in enumerate(u_s_l)}
    else:
        palette = list(secondary_colors.values())
        p_idx = 0
        for usl in u_s_l:
            if usl not in secondary_colors.keys():
                secondary_colors[usl] = palette[p_idx%len(palette)]
                p_idx += 1
        #
    #

    ####################

    fig,ax = pyplot.subplots(1,1)
#    fig.subplots_adjust(right=0.98,left=0.15)

    # Should these be optional arguments?
    # Or just pass **kwargs to ax.bar and hope for the best?
    #bar_centers = np.mean([])
    bar_width = 0.9*(bins[1] - bins[0])
    bar_edgecolor = [0,0,0,0.7]

    nc = len(bins) - 1
    # Reset active parameters
    current_height = np.zeros(nc)

    eq = {usl:np.where(usl==labels) for usl in u_s_l}

    for j,(k,v) in enumerate(eq.items()):
        # Get the secondary-class specific counts for the given
        # true label, across all predicted labels.
        counts,bin_edges = np.histogram(arr[v], bins=bins)
        bar_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        # Generate the portion of the bar plot given these counts.
        ax.bar( bar_centers,
                counts,
                bottom=current_height,
                color=secondary_colors[k],
                edgecolor=[bar_edgecolor for _ in bar_centers],
                width=bar_width
        )

        current_height += counts
    #
    #ax.set_xticks(bar_centers)
    #ax.set_xticklabels(cols)

#    ax.text(-0.08,0.5,str(tl), va='center',ha='right', transform=ax.transAxes)

    # Generate labels and a legend; skip if user didn't
    # specify secondary labels.
    if not skip_legend:
        for sv in u_s_l:
            ax.scatter([],[],marker='s',c=[secondary_colors[sv]],label=str(sv),s=80)
        #

        # Rudimentary automatic detection of stuff-in-the-way
        # for the legend location. ax does this automatically, but
        # can sometimes put legend in the center (which we don't want.)

        if np.argmax(current_height)==nc-1:
            ax.legend(loc='upper left')
        else:
            ax.legend(loc='upper right')
        #
    #

    ##############
    # Some tweaks
    ax.yaxis.grid(True)     # horizontal grid lines
    ax.set_axisbelow(True)  # grid lines beneath bars

    # detect order of magnitude to make "nice" yticks.
    maxy = max(1, np.max(current_height))
    oom = max(0, int( np.floor(np.log10(maxy)) ) )
    # Choose basic unit for steps based on data being <1 or >1.
    dy = 10**oom
    # Checks for too few or too many expected ticks.
    #
    # Note: by design, should have between 1 and 10 ticks, but
    # we'd like between ~3 and 5.
    if maxy/float(dy) > 5:
        dy = 2*dy
    elif maxy/float(dy) < 3 and dy!=1:
        dy = dy//2
    #
    # finally, set the ticks.
    if oom<0:
        yticks = np.arange(0, maxy+dy/2., dy)
    else:
        yticks = np.arange(0, int(maxy)+dy/2, int(dy))
    #
    ax.set_yticks(yticks)

    ####################

    #


    if kwargs.get('show',False):
        fig.show()
    #
    return fig,ax
#

if __name__=="__main__":
    import numpy as np
    n = 1000
    values = np.random.randn(n)
    labels = np.random.choice(['lions','tigers','bears','koalas'],n)

    fig,ax = stacked_hist(values,labels)
    pyplot.show(block=False)

#
