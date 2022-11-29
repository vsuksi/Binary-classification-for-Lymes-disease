'''
Purpose: provide a clean template to improve upon matplotlib.
Basic goals:
    - Ensure scatter plots, line plots, etc, are visible and legible
        under poor lighting (e.g. old projectors)
    - Ensure text sizes, legends, etc, are legible
        (can a Boomer read it?)
    - Provide a uniform visual style associated with calcom.

I'd like this to be an "easy to use" tool, meaning you should be able
to feed in a figure and axis and it simply "cleans it up" for you.
Whether this is feasible or not, I don't know.
'''
import matplotlib
from matplotlib import pyplot
from matplotlib import rcParams
import numpy as np

ref_cmap = pyplot.cm.tab10
nrefs = 8
prop_colors = [ref_cmap(j) for j in range(nrefs)]
prop_markers = ['o','P', 'X', 's']     # capital P,X are "fillable" plus/cross.
#prop_markersizes = [6,8,8,6]

# Note - pyplot scatter does **not**
# support cycling for anything but color right now, see
# https://github.com/matplotlib/matplotlib/issues/15105
# Temp workaround is to use plot() with 0 linewidth.... not great.
prop_cycle = matplotlib.rcsetup.cycler(
    color=np.tile( prop_colors, (nrefs//len(prop_colors)+1, 1) )[:nrefs],
    marker=np.tile( prop_markers, nrefs//len(prop_markers)+1 )[:nrefs]
#    markersize=np.tile( [6,8,8,6], nrefs//4+1 )[:nrefs]
)

######################
#
# dictionary entries to replace in rcParams.
ccParams = {
    # figure properties
    'figure.figsize': [10,8],
#    'figure.constrained_layout.use': True,

    # font-related things
    'font.family': 'monospace',
    'font.size': 14,
    'legend.fontsize': 14,

    # axis properties
    'grid.color': '#606060',
    'grid.linewidth': 0.5,
    'axes.prop_cycle': prop_cycle,

    # pyplot.plot shaping
    'lines.linewidth': 2,
    'lines.markeredgecolor': '#000000',
    'lines.markeredgewidth': 0.25,
    'lines.markersize': 5
}
##############

for k,v in ccParams.items():
    if k not in rcParams:
        print('Warning: key %s not found in default matplotlib.rcParams.'%k)
        print('This property might not be reflected in plots.')
    else:
        rcParams[k] = v
#


# todo: color scheme
# todo: font
# todo: font, plot, figure scaling

def apply_style(fig,ax):
    # apply the style defined at the top of this file:
    # colors, font

    pass
#

def rescale_fig(fig, mindim=8):

    figh = fig.get_figheight()
    figw = fig.get_figwidth()
#    mindim = min(figh,figw)

    if (figh <= figw) and (figh < mindim):
        figw *= mindim/figh
        figh = mindim
    elif (figh > figw) and (figw < mindim):
        figh *= mindim/figw
        figw = mindim
    #
    fig.set_figheight(figh)
    fig.set_figwidth(figw)
#

def clean_scatter(fig,ax):
    from numpy import sqrt
    rescale_fig(fig)

    # scale up the plot if needed.
    mindim = 8 #inches
    figh = fig.get_figheight()
    figw = fig.get_figwidth()

    figscale = min(figh,figw)

    # try 0: scatter sizes should be chosen
    # as a proportion of the minimum size of figure.
    colls = ax.collections
    npoints = sum( [len(coll.get_offsets()) for coll in colls] )

    scale_factor = 100. / sqrt(npoints)
    collections = ax.collections
    for coll in collections:
        coll.set_sizes([scale_factor*figscale])
    #

    # update legend font size.
    ax.legend(fontsize=18)

    fig.tight_layout()

    return
#
