
from calcom.classifiers._abstractclassifier import AbstractClassifier
# from calcom.metrics import ConfusionMatrix
# import numpy as np
# import multiprocessing
# bsr = ConfusionMatrix(return_measure='bsr')

# Unsure if this is the best idea, or if referencing
# TreeClassifier() is better. Doing the latter would take
# some more work to fit in to existing code.
# from calcom.classifiers.treeclassifier import Tree

class RFClassifier(AbstractClassifier):
    '''

    An class implementing a random forest algorithm.
    Random forests are trained by feeding
    random samples of the data, and random subspaces thereof,
    to a "large" number of classification trees (also
    implemented independently in calcom). The training
    is done by training these classification trees.

    The classification on new data is done by feeding
    it appropriately to all the classification trees,
    which then each provide a vote. The overall
    classification is then based on the majority vote
    of the trees.
    '''

    def __init__(self):
        '''
        Initialize the classification tree. Tunable parameters
        are the same as those in the Tree() class and
        are passed by reference (so only self.params needs to be changed).
        '''
        self.forest = Forest()
        self.params = {}
        self.results = {}

        self.params['ntrees'] = 10
        self.params['subset_prop'] = 0.1
        self.params['subdim_prop'] = 0.1
        self.params['nprocs'] = 1

        # If True, returns proportion of votes for the second element
        # in np.unique(labels). Useful for visualizing.
        self.params['return_prob'] = False

        super().__init__()

    #

    @property
    def _is_native_multiclass(self):
        return True
    #
    @property
    def _is_ensemble_method(self):
        return False

    def _fit(self,data,labels):

        # A bit of an annoyance; parameters
        # need to be passed to the Forest class here in case
        # the user changes the parameters before fitting.

        # internal_labels = self._process_input_labels(labels)

        self.forest.return_prob = self.params['return_prob']
        self.forest.ntrees = self.params['ntrees']
        self.forest.subset_prop = self.params['subset_prop']
        self.forest.subdim_prop = self.params['subdim_prop']
        self.forest.nprocs = self.params['nprocs']


        # self.forest.load_data(data,internal_labels)
        self.forest.load_data(data,labels)
        self.forest.plant_trees()
        self.forest.grow_trees()

        return
    #

    def _predict(self,data):
        pred_labels_internal = self.forest.classify(data)
        # pred_labels = self._process_output_labels(pred_labels_internal)
        # return pred_labels
        return pred_labels_internal
    #

    def visualize(self,*args):
        pass
    #

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

    def isolate_subsets(self, data, labels):
        '''
        Given a trained random forest, and a data set with labels,
        returns:
            (0) A percent correct classification of the trees on the data;
            (1) A list indicating a ranking of the
            classification trees based on total correct votes of
            labels (best to worst) (i.e., based on np.argsort() of the first list)
            (2) A list indicating a scoring of dimensions based on
            a sum over the trees, weighted by probabilities from (0),
            summed over every split, and by depth of the split in the tree.
            (3) np.argsort() of the negative of the result of (2); that is, a
            heuristic ranking of the dimensions from most important to least.
        '''
        from calcom.metrics import ConfusionMatrix
        import numpy as np

        # import multiprocessing
        bsr = ConfusionMatrix(return_measure='bsr')


        treescores = np.zeros(self.params['ntrees'])
        dimscores = np.zeros(np.shape(data)[1])

        alpha = 0. # Weight favoring (inverse) tree size to treescore (bsr). alpha=0 -> only accounts for tree size.

        for i,tree in enumerate(self.forest.trees):
            pred_labels = tree.classify(data)
            # treescores[i] = np.sum(pred_labels==labels)/len(labels)
            treescores[i] = bsr.evaluate(labels,pred_labels) - 0.5 # Subtract "random guessing" score
            dims = self.forest.tree_subspaces[i]
            dimscores[dims] += alpha*treescores[i] + (1.-alpha)*1./(1+len(tree.tree))
        #
        tree_ranking = np.argsort(-treescores)

        # dimscores = np.zeros(np.shape(data)[1])
        # for i,tree in enumerate(self.forest.trees):
        #     dims = self.forest.tree_subspaces[i]
        #     splitscores = np.array( [ (not node.isLeaf)/(2.**node.depth) for node in tree.tree] )
        #     splitscores /= np.sum(splitscores)
        #
        #     for j,node in enumerate(tree.tree):
        #         if (not node.isLeaf):
        #             dimscores[ dims[node.decision[0]] ] += treescores[i] * splitscores[j]
        #         #
        #     #
        # #
        dim_ranking = np.argsort(-dimscores)

        return treescores,tree_ranking,dimscores,dim_ranking
    #

#

class Forest(object):

    def __init__(self):

        # NOTE: These will be overwritten when RFClassifier is initialized!

        self.nprocs = 4             # Number of processes for parallelization
        self.return_prob = True
        self.ntrees = 100

        self.data = []
        self.labels = []
        self.trees = []
        self.subset_prop = 0.1      # Proportion of data to use for trees.
        self.subdim_prop = 0.1        # Proportion of number of dimensions to use for trees.

        # Calculated once data is loaded.
        self.nvecs = 0
        self.nsubvecs = 0
        self.ndims = 0
        self.nsubdims = 0
        self.unique_labels = []
        self.labelmap = {}
        self.invlabelmap = {}

        # Calculated when the forest is planted.
        # Needs to be saved to do classifications after training.
        self.tree_subspaces = []
        self.label_subspaces = []
    #

    def load_data(self,data,labels):
        '''
        Places the input arrays "data" and "labels"
        into self.data and self.labels. No error checking
        is done. The array labels is assumed to have {0,1} labels.
        '''
        import numpy as np

        self.data = data
        self.labels = labels

        self.nvecs,self.ndims = np.shape(data)
        self.nsubvecs = max(1, int( self.nvecs * self.subset_prop ))
        self.nsubdims = max(1, int( self.ndims * self.subdim_prop ))

        self.unique_labels = np.unique(self.labels)

        # Mapping of the labels to the integers for polling.
        self.labelmap = {}
        self.invlabelmap = {}
        for i,lab in enumerate(self.unique_labels):
            self.labelmap[lab] = i
            self.invlabelmap[i] = lab
        #
    #

    def plant_trees(self):
        '''
        Sets up the ensemble with the paremeters in self.
        '''
        from calcom.classifiers._treeclassifier import Tree
        import numpy as np

        for i in range(self.ntrees):
            self.trees.append(Tree())

            # Feed a random subset and subspace of data to the tree.
            vec_idxs = np.random.permutation(range(self.nvecs))[:self.nsubvecs]
            dim_idxs = np.random.permutation(range(self.ndims))[:self.nsubdims]

            # Need to keep track of appropriate indexes to pass to each tree
            # for classification.
            self.tree_subspaces.append( dim_idxs )
            self.label_subspaces.append( vec_idxs )

            subdata = self.data[vec_idxs,:]
            subspacedata = subdata[:,dim_idxs]

            self.trees[i].load_data(subspacedata , self.labels[vec_idxs] )
        #
    #

    def grow_trees(self):
        '''
        Once everything is set up, train all the trees.
        '''
        import multiprocessing
        import numpy as np

        # Experimenting with multiprocessing.

        pool = multiprocessing.Pool(self.nprocs)

        # print(self.trees[0].tree)

        # async_result = pool.map_async( grow_wrapper, self.trees )
        # self.trees = async_result.get()
        self.trees = pool.map( grow_wrapper, self.trees )

        pool.close()
        return
    #

    def classify_datum(self,datum):
        '''
        Classify a single datum.

        This now can handle either (a) a majority vote amongst all trees,
        or (b) the ratio voting for the self.unique_labels[1],
        based on whether self.return_prob=False or
        self.return_prob=True (respectively).
        '''
        import numpy as np

        # This eats memory like crazy - need to figure out how to properly
        # structure code to avoid this.

        # allthings = [ [ self.trees[i], datum, self.tree_subspaces[i] ] for i in range(self.ntrees) ]
        #
        # pool = multiprocessing.Pool(self.nprocs)
        # async_result = pool.map_async( cd_wrapper, allthings )
        # votes = async_result.get()
        # pool.close()
        #
        # # votes = pool.map( cd_wrapper, allthings )
        #
        # tally = np.zeros(np.shape(self.unique_labels))
        #
        # for i in range(self.ntrees):
        #     tally[ self.labelmap[votes[i]] ] += 1
        # #

        tally = np.zeros(np.shape(self.unique_labels))
        for i,tree in enumerate(self.trees):
            didxs = self.tree_subspaces[i]
            # tally[ self.labelmap[tree.classify_datum(datum[didxs])] ] += 1
            # print(didxs)
            # print(datum[didxs])
            # print(tree.classify_datum(datum[didxs]))
            # print(self.labelmap[tree.classify_datum(datum[didxs])])
            tally[ self.labelmap[tree.classify_datum(datum[didxs])] ] += 1
        #

        if self.return_prob:
            val = float(tally[1])/np.sum(tally)
            return val
        else:
            return self.invlabelmap[np.argmax(tally)]
        #
    #

    def classify(self,data):
        '''
        Iteration of classify_datum over all the data.
        '''

        pred_labels = []

        for i,datum in enumerate(data):
            # Right now this is a manual toggle between discrete and
            # continuum values. Might split this into two methods later.
            # Note that the return statement of self.classify_datum

            pred_labels.append( self.classify_datum(datum) )
            # pred_labels.append( self.classify_datum_real(datum) )
        #
        # pred_labels = np.array( pred_labels )

        return pred_labels
    #

    def get_treeleaf_distribution(self):
        '''
        Returns a pair of arrays size,count containing the
        distribution of number of tree leaves.
        Useful for visualizing with pyplot.bar.
        '''
        import numpy as np

        size = []
        count = []
        sizemap = {}
        for tree in self.trees:
            nl = len(tree.collect_leaves())

            if nl not in size:
                sizemap[nl] = len(size)
                size.append(nl)
                count.append(0)
            #
            count[sizemap[nl]] += 1
        #
        idxs = np.argsort(size)

        size = np.array(size)[idxs]
        count = np.array(count)[idxs]

        return size,count
    #
#

# Testing
if __name__ == "__main__":

    from matplotlib import pyplot
    import artificial_distributions as ad

    # Reset the RNG and set parameters.
    ad.reset_seed()
    ntrees = 400
    npoints = 200
    ng = 21
    mu = [2,-2]

    # Create train/test datasets.
    tr_data,tr_labels = ad.point_clouds_2d(mu=mu)
    bounds = [tr_data[:,0].min(), tr_data[:,0].max(), tr_data[:,1].min(), tr_data[:,1].max()]
    xg,yg,te_data = ad.grid_2d(bounds,ng)

    # Set up the classification.
    mo = Forest()
    mo.load_data(tr_data,tr_labels)
    mo.ntrees = ntrees
    print("Setting up forest...")
    mo.plant_trees()
    print("Training trees...")
    mo.grow_trees()

    print("Visualizing one of the trees...")
    tree = mo.trees[0]
    fig,ax,coords,col,adj = tree.visualize_tree()

    pyplot.show(block=False)
#

# Wrapper functions. Having issues including them inside the
# classes, or even nested inside another function.

def grow_wrapper(tree):
    '''
    Wrapper for tree.grow() to be used with pool.map()
    with the multiprocessing package.
    '''
    tree.grow()
    return tree
#

def cd_wrapper(things):
    tree,datum,didxs = things
    return tree.classify_datum(datum[didxs])
#
