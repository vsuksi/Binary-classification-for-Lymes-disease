#from __future__ import absolute_import, division, print_function
from calcom.metrics._abstractmetric import AbstractMetric

class ConfusionMatrix(AbstractMetric):

    def __radd__(self,other):
        '''
        Does the same thing as __add__().
        Apparently needs to be defined for sum() to work.
        '''
        if type(other)==int:
            # python sum() begins by trying to add 0 to the first element;
            # just return the thing itself.
            return self
        else:
            return other.__add__(self)  # to preserve expected ordering of labels.
        #
    #

    def __add__(self,other):
        '''
        Overloading the addition operation, so that
        we can "easily" add together multiple confusion
        matrices. Essentially what happens is:

        1. The true/predicted labels in two arrays are concatenated;
        2. Statistics are recalculated using the self.evaluate() function.

        No error checking is done right now. Who knows what
        will happen if you try to break this!
        '''
        nc = len(self.results['cf'])

        cm_new = ConfusionMatrix()
        y_true_new = list(self.params['true_labels']) + list(other.params['true_labels'])
        y_pred_new = list(self.params['pred_labels']) + list(other.params['pred_labels'])
        _ = cm_new.evaluate(y_true_new, y_pred_new)

        return cm_new
    #

    def __init__(self, return_measure = None):
        '''
        Setup default parameters
        Inputs:
            return_measure (default=None): if specified will return that measure from evaluate function
                options are:    tpr: True positive rate
                                tnr: True negative rate
                                ppv: Positive predictive value/precision
                                npv: Negative predicitve value
                                fnr: False negative rate/miss rate
                                fpr: False positive rate/fall-out
                                fdr: False discovery rate
                                for: False omission rate
                                acc: Accuracy
                                bsr: Balanced success rate
        '''
        self.return_measure = return_measure
        self.params = {}
        self.params['bsrmode'] = "default"
        self.params['bw'] = 0
        self.params['blocks'] = None

        self.results = {}

        self.results['cf'] = None

        # See https://en.wikipedia.org/w/index.php?title=Confusion_matrix&oldid=794367294
        self.results['tpr'] = None # True positive rate
        self.results['tnr'] = None # True negative rate
        self.results['ppv'] = None # Positive predictive value/precision
        self.results['npv'] = None # Negative predicitve value
        self.results['fnr'] = None # False negative rate/miss rate
        self.results['fpr'] = None # False positive rate/fall-out
        self.results['fdr'] = None # False discovery rate
        self.results['for'] = None # False omission rate
        self.results['acc'] = None # Accuracy
        self.results['bsr'] = None # Balanced success rate

        self.params['pred_labels'] = []
        self.params['true_labels'] = []
    #

    def evaluate(self, y_true, y_pred, labels=None):
        '''
        Create a confusion matrix, given a list or array
        of true labels and predicted labels as the result
        of a binary classifier.

        Inputs:
            y_true : array, shape = [n_samples]
                Ground truth (correct) target values.

            y_pred : array, shape = [n_samples]
                Estimated targets as returned by a classifier.

            labels : array, shape = [n_classes], optional
                List of labels to index the matrix. This may be used to reorder
                or select a subset of labels.
                If none is given, those that appear at least once
                in ``y_true`` or ``y_pred`` are used in sorted order.

        Optional parameters:
            bsrmode : string
                One of "default", "bandwidth", or "block", changing how the
                BSR is calculated from a given confusion matrix:

                    "default" - Default BSR; simple average of the success rates
                        for each of the labels, done by normalizing each row
                        by its sum, taking the trace, and dividing by the
                        size of the matrix.
                    "bandwidth" - Also requires the bw, an integer input
                        specifying the size of the bandwidth. The confusion
                        matrix is row-normalized, all entries within bw of the
                        main diagonal are summed, and then divided by the size
                        of the matrix.
                    "block" - Also requires blocks input, which can be a
                        dictonary mapping original labels to their new labels,
                        or a list of lists, with the elements being a disjoint
                        partition of the original labels. A default BSR is
                        calculated with these new labels.
                Note that this is currently only implemented for BSR. It may
                make sense to be implemented for other metrics in the future,
                but it isn't a high priority.

        Outputs:
                C : array, shape = [n_classes, n_classes]
                Confusion matrix

            See https://en.wikipedia.org/w/index.php?title=Confusion_matrix&oldid=794367294
            for the definitions of the remaining metrics.

        '''
        import numpy as np
        import warnings


        # Verify that the predicted labels are a subset
        # of the actual labels.
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()

        # if (not isSubset(y_pred,y_true)):
        #     raise ValueError("The predicted labels are inconsistent with the actual labels. ConfusionMatrix.evaluate cannot continue.")
        # #

        an = len(y_true)
        pn = len(y_pred)

        if (an!=pn):
            raise ValueError("Predicted and actual label arrays are different sizes. ConfusionMatrix.evaluate cannot continue.")
        #

        if labels is None:
            # labels = np.sort(np.unique(y_true))
            labels = np.sort(np.union1d( y_true, y_pred ))
        else:
            labels = np.asarray(labels)
            if np.all([l not in y_true for l in labels]):
                raise ValueError("At least one label specified must be in y_true")
        #

        self.params['true_labels'] = y_true
        self.params['pred_labels'] = y_pred

        # n_labels = labels.size
        n_labels = len(labels)


        label_to_ind = dict((y, x) for x, y in enumerate(labels))

        # convert yt, yp into index
        y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
        y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

        # intersect y_pred, y_true with labels, eliminate items not in labels
        ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
        y_pred = y_pred[ind]
        y_true = y_true[ind]


        confmat = np.zeros( (n_labels,n_labels) ,dtype='i8')

        for k in range(an):
            confmat[y_true[k],y_pred[k]] += 1
        #

        self.results['cf'] = confmat

        # correct predictions
        diagonal = confmat.diagonal()

        total = np.sum(confmat)
        correct = np.sum(diagonal)
        self.results['acc'] = float(correct)/(total)

        if self.params['bsrmode']=="default":
            self.results['bsr'] =0

            n_true = 0 # count number of unique true labels in confusion matrix.
            for c in range(n_labels):
                correct_preds = confmat[c,c]
                total_c = confmat[c,:]
                true_count = np.sum(total_c)
                if true_count>0:
                    self.results['bsr'] += float(correct_preds)/true_count
                    n_true += 1
                #
            #
            # self.results['bsr'] /= n_labels
            self.results['bsr'] /= n_true
        elif self.params['bsrmode']=="bandwidth":
            bwparam = self.params['bw']
            if type(bwparam)!=list:
                # Assume a positive integer (real number is okay too),
                # and make a list doing +/- the number.
                bw = [-abs(bwparam),abs(bwparam)]
            else:
                bw = list(bwparam)
            #
            classscores = np.zeros(n_labels)
            for i in range(n_labels):
                left = max(i+bw[0],0)
                right = min(i+bw[1]+1,n_labels)
                classscores[i] = sum(confmat[i,left:right])/sum(confmat[i,:])
            #
            self.results['bsr'] = sum(classscores)/float(n_labels)
        elif self.params['bsrmode']=="block":
            # blocks = kwargs.get("blocks", {i:i for i in range(n_labels)} )
            if type(self.params['blocks'])==dict:
                y_true_blocked = [ self.params['blocks'][y_t] for y_t in y_true ]
                y_pred_blocked = [ self.params['blocks'][y_p] for y_p in y_pred ]
            elif type(self.params['blocks'])==list:
                # Just make the dictionary here and copy the code above.
                blocks_dict = {}
                for i,part in enumerate(self.params['blocks']):
                    for elem in part:
                        blocks_dict[elem] = i
                    #
                #
                y_true_blocked = [ blocks_dict[y_t] for y_t in y_true ]
                y_pred_blocked = [ blocks_dict[y_p] for y_p in y_pred ]
            #
            cf2 = ConfusionMatrix(return_measure='bsr')
            self.results['bsr'] = cf2.evaluate(y_true_blocked, y_pred_blocked)
        #

        if n_labels == 2:
            # maybe in multi-class case these can be arrays. i.e. 'tpr' for each class.
            tp,fn,fp,tn = confmat.flatten()
            p = tp + fn
            n = tn + fp
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    self.results['tpr'] = float(tp)/p
                    self.results['tnr'] = float(tn)/n
                    self.results['ppv'] = float(tp)/(tp+fp)
                    self.results['npv'] = float(tn)/(tn+fn)
                    self.results['fnr'] = float(fn)/p
                    self.results['fpr'] = float(fp)/n
                    self.results['fdr'] = float(fp)/(fp+tp)
                    self.results['for'] = float(fn)/(fn+tn)
                except ZeroDivisionError:
                    pass


        if self.return_measure in self.results:
            return self.results[self.return_measure]
        else:
            return confmat
    #

    def visualize(self, type='heatmap', **kwargs):
        '''
        A parent function which allows for switching between
        several different types of visualization supported.

        Inputs: None

        Optional inputs:
            type : string. Specifies the function to use:
                'heatmap' : self.visualize_bsr(**kwargs)
                'barplot' : self.visualize_barplot(**kwargs)
                All optional arguments are passed to the
                corresponding function.

        Outputs: a fig,ax pair corresponding to the visualization
            requested.
        '''
        switch_dict = {
            'heatmap': self.visualize_bsr,
            'barplot': self.visualize_barplot
        }

        if type not in switch_dict.keys():
            keys_str = ', '.join(switch_dict.keys())
            raise ValueError('Visualization type %s not recognized. Please choose between %s.'%( str(type), keys_str ) )
        #

        return switch_dict[type](**kwargs)
    #

    def find_misclassified(self,true_value,pred_value):
        '''
        After self.evaluate() has been called,
        '''
        pass
    #

    def visualize_bsr(self, **kwargs):
        '''
        Visualize a confusion matrix given the setings specified in
        self.params.

        Required inputs are expected in one of three forms:

            1. If no arguments are specified, self.params['cf']
               is used as a confusion matrix. Visualization is done using that.
            2. If arguments y_true=... and y_pred=... are specified, then
               self.evaluate() is called, then that confusion matrix is used.
            3. If argument cf=... is specified, this is assumed a confusion
               matrix and is used.

            Another input row_normalize indicates whether to visualize the
            row-normalized confusion matrix. Defaults to True. In the case
            self.params['bsrmode'] = 'block', the "normalization" is done
            using the blocks, in the same manner that the BSR would be calculated.

        Once the confusion matrix is obtained, visualization is done using
        the settings specified in self.params['bsrmode'], self.params['bw'],
        and self.params['blocks'].

        Optional inputs (other than discussed above):
            show : Boolean; whether to show the plot immediately. Default: False

        Outputs:
            pyplot figure, axis handle for the plot, and axis handle for the
            colorbar. A call to pyplot.show() will show the result.

        Examples:
            1. Using a matrix as an input:
            -----------------------------------------------------------------
            from calcom.metrics import ConfusionMatrix()
            from matplotlib import pyplot

            confmat = np.array([[4,0,0], [1,3,0], [2,1,5]])
            cf = ConfusionMatrix()
            fig,ax = cf.visualize_bsr(cf=confmat)
            pyplot.show(block=False)

            2. Using labels as inputs; using block BSR mode:
            -----------------------------------------------------------------
            from calcom.metrics import ConfusionMatrix()
            from matplotlib import pyplot

            y_true = [0,0,0,0,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,3]
            y_pred = [0,0,0,1,0,1,2,1,2,2,3,0,0,1,1,2,3,3,3,3,3]

            cf = ConfusionMatrix()
            cf.params['bsrmode'] = 'block'
            cf.params['blocks'] = [[0,1], [2,3]]

            fig,ax = cf.visualize_bsr(y_true=y_true, y_pred=y_pred)
            pyplot.show(block=False)

            3. Using previously calculated input; using bandwidth BSR mode:
            -----------------------------------------------------------------
            from calcom.metrics import ConfusionMatrix()
            from matplotlib import pyplot

            # Previous code goes here.

            cf.params['bsrmode'] = 'bandwidth'
            cf.params['bw'] = 3

            fig,ax = cf.visualize_bsr()
            pyplot.show(block=False)
        '''

        from matplotlib import pyplot
        import numpy as np

        # Read input and class parameters.
        y_true = kwargs.get('y_true',[])
        y_pred = kwargs.get('y_pred',[])
        row_normalize = kwargs.get('row_normalize',True)

        # Get the confusion matrix we'll be working with.
        # Note, if y_true and y_pred are provided, it will overwrite anything
        # already in self.params['cf'].
        if len(y_true)>0 and len(y_pred)>0:
            self.evaluate(y_true,y_pred)
            cf = self.results['cf']
        #
        cf = kwargs.get('cf', self.results['cf'])


        n,_ = cf.shape


        mode = self.params['bsrmode']

        # For each mode, do the appropriate stuff.
        # Get the outline of the selected portion
        if mode=="default":
            # Row-normalized if requested.
            if row_normalize:
                rowsums = np.sum(cf,1)
                cf = np.dot( np.diag(1./rowsums), cf )
            #
            outline = np.eye(n, dtype=int)
            bsrval = np.sum(np.diag(cf))/n
        elif mode=="bandwidth":

            bwparam = self.params['bw']
            if type(bwparam)==int:
                bw = [-abs(bwparam),abs(bwparam)]
            else:
                bw = list(bwparam)
            #

            # Row-normalized if requested.
            if row_normalize:
                rowsums = np.sum(cf,1)
                cf = np.dot( np.diag(1./rowsums), cf )
            #

            # No error checking/clipping done here! Careful!
            bsrval = np.sum( [ np.sum(np.diag(cf,k)) for k in range(bw[0],bw[1]+1) ] )/n

            outline = np.zeros( (n,n), dtype=int)
            for i in range(n):
                left = max(0,i+bw[0])
                right = min(n,i+bw[1]+1)
                outline[i,left:right] = 1
            #
        elif mode=="block":
            blocks = self.params['blocks']

            if type(blocks)==dict:
                # Convert to list of lists. Not the prettiest but it works.
                blocks_list = []
                indices = {}

                for key in blocks.keys():
                    if blocks[key] in indices:
                        blocks_list[indices[blocks[key]]].append( key )
                    else:
                        indices[blocks[key]] = len(blocks_list)
                        blocks_list.append([key])
                        # blocks_list[indices[blocks[key]]].append( key )
                    #
                #
            else:
                # Assume it's already a dictionary.
                blocks_list = list(blocks)
            #


            rowsums = np.zeros(n)
            blockrowsums = [0 for block in blocks_list]

            for i,block in enumerate(blocks_list):
                for element in block:
                    blockrowsums[i] += np.sum(cf[element,:])
                #
            #
            for i,block in enumerate(blocks_list):
                for element in block:
                    rowsums[element] = blockrowsums[i]
                #
            #

            # Row-normalized if requested.
            if row_normalize:
                cf = np.dot( np.diag(1./rowsums), cf )
            #

            # Calculate BSR
            bsrval = 0.
            for block in blocks_list:
                bsrval += np.sum( [ cf[elem,block] for elem in block ] )
            #
            bsrval /= len(blocks)

            outline = np.zeros( (n,n), dtype=int)
            for block in blocks_list:
                for i in block:
                    for j in block:
                        outline[i,j] = 1
                    #
                #
            #
        #

        # Ugly hack to make ax.contour do what we want; unnecessary memory, etc.
        # In future, just implement something to work with the original
        # outline matrix. pyplot.contour, pyplot.pcolor, etc, don't really
        # perform the way we want.
        ssf = 101
        n_ss = ssf*n
        outline_ss = np.zeros( (n_ss,n_ss), dtype=int )
        ii,jj = np.where(outline==0)

        for k in range(len(ii)):
            outline_ss[ssf*ii[k]:ssf*(ii[k]+1), ssf*jj[k]:ssf*(jj[k]+1)] = np.ones( (ssf,ssf), dtype=int )
        #
        xx,yy = np.meshgrid( np.linspace(0,n,n_ss), np.linspace(0,n,n_ss) )

        ###############

        # Start setting up the plot.

        fig,ax = pyplot.subplots(1,1)

        # mycm = pyplot.cm.Greys
        mycm = pyplot.cm.viridis
        im = ax.pcolor(cf, vmin=0., vmax=cf.max(), cmap=mycm)
        # ax.pcolor(xx,yy,outline_ss, cmap=pyplot.cm.Reds, alpha=0.2)
        # ax.contourf(xx,yy,outline_ss, levels=[0,0.9], colors=[[1,0,0,1],[0.2,0.2,0.2,1]], alpha=0.05)

        ###########
        # For shading the regions used. Not happy with this right now.
        # from copy import copy
        # mycm = copy(pyplot.cm.Reds)
        # mycm.set_under('r',0.)
        # ax.imshow(outline, cmap=mycm, vmin=0.1, vmax=1, interpolation='nearest', extent=(0,n,0,n), origin='lower', alpha=0.1, zorder=1000)

        ax.contour(xx,yy,outline_ss, levels=[0.5], colors='r')

        # Add a colorbar
        fig.colorbar(im, ax=ax)
        im.set_clim(0., cf.max())

        ax.axis('square')
        ax.set_xlim([0,n])
        ax.set_ylim([0,n])
        ax.invert_yaxis()
        ax.set_xlabel('Predicted label', fontsize=14)
        ax.set_ylabel('True label', fontsize=14)
        ax.set_xticks(0.5+np.arange(n))
        ax.set_yticks(0.5+np.arange(n))
        ax.set_xticklabels(np.arange(n,dtype=int))
        ax.set_yticklabels(np.arange(n,dtype=int))

        if mode=="default":
            title = "Simple BSR: %.3f"%bsrval
        elif mode=="bandwidth":
            title = "Banded BSR: %.3f"%bsrval
        elif mode=="block":
            title = "Block BSR: %.3f"%bsrval
        #

        ax.set_title(title,fontsize=18)

        if kwargs.get('show',False):
            fig.show()
        #

        return fig,ax
    #

    def visualize_barplot(self, **kwargs):
        '''
        Visualizes the confusion matrix as a sequence of bar plots.
        May not scale well to a large number of classes.
        Assumes the essential data has been populated by using a call
        to self.evaluate().

        Inputs: None
        Optional inputs:
            secondary_labels : iterable of the same size as
                self.params['true_labels']. If specified, a
                stacked barplot is generated which visualizes
                how quantities of a secondary label are assigned.
                The entries may be of any type, and are labeled
                in the figure. Default: None

            bar_colors : Either a single tuple, to use a single color for all bars,
                or a dictionary mapping secondary label
                names to colors; internally passed to pyplot.bar.
                If dictionary entries do not exist for required labels,
                colors from pyplot.cm.Set3 are sampled.

            rows : If specified, a subset of rows of the confusion matrix
                is actually visualized. If an iterable, the corresponding
                classes are displayed. Else, only the specified true class
                results are displayed. This is indexed BY ATTRIBUTE LABEL.
                Default: all classes.

            cols : If specified, the given predicted labels are plotted
                _in the given order_. This will override the default
                ordering, which is automatically sorted by label name.

            which : Alias for 'rows' for (limited) backward compatibility.

            show : Boolean; whether to show the plot immediately. Default: False

        Outputs: pyplot figure, axis pair.
        '''
        from matplotlib import pyplot
        from calcom.utils import type_functions as tf
        import numpy as np

        # Process known data; generate defaults if needed.
        confmat = self.results['cf']
        pred_labels = self.params['pred_labels']
        true_labels = self.params['true_labels']

        nc = confmat.shape[0]

        u_l = np.unique(true_labels)

        nc = len(u_l)

        secondary_labels = kwargs.get('secondary_labels', [])
        if len(secondary_labels) == 0:
            # Flag to remind ourselves not to later create a legend
            # when one wasn't desired in the first place.
            skip_legend = True

            # Create nominal secondary labels for the loop below.
            secondary_labels = [0 for _ in true_labels]
        else:
            skip_legend = False
        #

        secondary_labels = np.array(secondary_labels)
        u_s_l = np.unique(secondary_labels)

        # THIS SWITCH WILL BE REMOVED IN THE FUTURE!
        if 'which' in kwargs.keys():
            classes_tovis = kwargs.get('which')
        else:
            classes_tovis = kwargs.get('rows', np.unique(true_labels))
        #

        cols = kwargs.get('cols', np.unique(true_labels))

        if not tf.is_list_like(classes_tovis):
            classes_tovis = [classes_tovis]
        #
        nc_tovis = len(classes_tovis)

        secondary_colors = kwargs.get('bar_colors', {})

        palette = pyplot.cm.Set3(np.linspace(0,1,12))   # pastel colors
        palette = palette[np.mod(np.arange(3,len(palette)+3),len(palette))] #reorder for good starting colors

        if not type(secondary_colors)==dict:
            secondary_colors = {usl: secondary_colors for usl in u_s_l}
        else:
            p_idx = 0
            for usl in u_s_l:
                if usl not in secondary_colors.keys():
                    secondary_colors[usl] = palette[p_idx%len(palette)]
                    p_idx += 1
            #
        #


        ####################

        fig,ax = pyplot.subplots(nc_tovis,1, sharex=True)
        fig.subplots_adjust(right=0.98,left=0.15)

        # Hacky: if nc_tovis==1, pyplot.subplots() won't create a list
        # of axes, but we want to use the same code.
        # Create a purely internal list to handle this case.
        axl = [ax] if nc_tovis==1 else ax

        # Should these be optional arguments?
        # Or just pass **kwargs to ax.bar and hope for the best?
        bar_centers = np.arange(nc)
        bar_width = 0.9
        bar_edgecolor = [0,0,0,0.7]

        for k,tl in enumerate(classes_tovis):

            # Reset active parameters
            current_height = np.zeros(nc)

            for j,sv in enumerate(u_s_l):
                # Get the secondary-class specific counts for the given
                # true label, across all predicted labels.
                counts = []
                #for al in u_l:
                for al in cols:
                    confmat_row = np.logical_and( pred_labels==al, true_labels==tl  )
                    cf_row_secondary = np.logical_and(confmat_row, secondary_labels==sv)
                    counts.append( sum(cf_row_secondary) )
                #

                # Generate the portion of the bar plot given these counts.
                axl[k].bar( bar_centers,
                        counts,
                        bottom=current_height,
                        color=secondary_colors[sv],
                        edgecolor=[bar_edgecolor for _ in bar_centers],
                        width=bar_width
                )

                current_height += np.array(counts)
            #
            axl[k].set_xticks(bar_centers)
            axl[k].set_xticklabels(cols)
            # axl[k].set_yticks([current_height.max()/2.])
            # axl[k].set_yticklabels([str(tl)])
            # axl[k].set_yticks([])
            axl[k].text(-0.08,0.5,str(tl), va='center',ha='right', transform=axl[k].transAxes)

            # Generate labels and a legend; skip if user didn't
            # specify secondary labels.
            if not skip_legend:
                for sv in u_s_l:
                    axl[k].scatter([],[],marker='s',c=secondary_colors[sv],label=str(sv),s=80)
                #

                # Rudimentary automatic detection of stuff-in-the-way
                # for the legend location. ax does this automatically, but
                # can sometimes put legend in the center (which we don't want.)
                if k==0:
                    if np.argmax(current_height)==nc-1:
                        axl[k].legend(loc='upper left')
                    else:
                        axl[k].legend(loc='upper right')
                #
            #

            ##############
            # Some tweaks
            axl[k].yaxis.grid(True)     # horizontal grid lines
            axl[k].set_axisbelow(True)  # grid lines beneath bars

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
            axl[k].set_yticks(yticks)

            ####################

        #


        if kwargs.get('show',False):
            fig.show()
        #
        return fig,ax
    #
#

# def isSubset(arr1,arr2):
#     '''
#     Looks at two one-dimensional numpy arrays arr1 and arr2
#     and checks that the elements of np.unique(arr1) are in the
#     set np.unique(arr2). Returns True if this is the case,
#     else False.
#     '''
#     from numpy import unique
#     a_labset = unique(arr1)
#     p_labset = unique(arr2)
#     for elem in p_labset:
#         if all(elem!=p_labset):
#             return False
#         #
#     #
#     return True
# #

if __name__ == "__main__":
    # Testing the BSR modes.
    from matplotlib import pyplot

    tl = [0,0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4]
    pl = [0,0,1,2,1,1,1,0,3,4,0,0,3,3,3,3,4,0,1,2,3,4]
    cf = ConfusionMatrix()

    cf.params['bsrmode'] = 'default'    # Not necessary to specify this.
    cf.evaluate(tl,pl)
    bsr0 = cf.params['bsr']

    cf.params['bsrmode'] = 'bandwidth'
    bw = 1

    cf.params['bw'] = bw
    cf.evaluate(tl,pl)
    bsr1 = cf.params['bsr']

    cf.params['bsrmode'] = 'block'
    blocks = [[0,1,2],[3,4]]

    cf.params['blocks'] = blocks
    cf.evaluate(tl,pl)
    bsr2 = cf.params['bsr']

    print("Bandwidth:")
    print(bw)
    print("Blocks:")
    print(blocks)
    print("\n")
    print("True labels")
    print(tl)
    print("Pred labels")
    print(pl)
    print("\n")
    print("Confusion matrix:")
    print(cf.params['cf'])
    print("\n")

    print("BSRs: %.4f (default), %.4f (bandwidth), %.4f (block)"%(bsr0,bsr1,bsr2))
    print("Expected results: 0.4543, 0.6395, 0.6917")

    ##################
    # A few examples on how to use this.
    #
    cf.params['bsrmode'] = 'default'
    fig0,ax0 = cf.visualize_bsr()

    cf.params['bsrmode'] = 'bandwidth'
    cf.params['bw'] = 1
    fig1,ax1 = cf.visualize_bsr()

    cf.params['bsrmode'] = 'block'
    cf.params['blocks'] = [[0,1,2],[3,4]]
    fig2,ax2 = cf.visualize_bsr()

    pyplot.show(block=False)
#
