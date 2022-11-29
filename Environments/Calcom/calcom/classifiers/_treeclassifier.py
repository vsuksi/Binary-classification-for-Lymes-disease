# import numpy as np
from calcom.classifiers._abstractclassifier import AbstractClassifier

class TreeClassifier(AbstractClassifier):

    def __init__(self):
        '''
        Initialize the classification tree. Tunable parameters
        are the same as those in the Tree() class and
        are passed by reference (so only self.params needs to be changed).
        '''
        self.Tree = Tree()
        self.params = {}
        self.results = {}
        # self.params['']

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
        # internal_labels = self._process_labels(labels)
        # self.Tree.load_data(data,internal_labels)
        self.Tree.load_data(data,labels)
        self.Tree.grow()

    #

    def _predict(self,data):
        pred_labels_internal = self.Tree.classify(data)
        # self.results['pred_labels'] = pred_labels

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
#

class Node(object):
    def __init__(self):
        self.idx = None                 # Node's index.
        self.children = []              # Pointers to child nodes
        self.parent = None              # Pointer to parent node
        self.majority_label = None      # Majority label for all points in region.
        self.mcr = None                 # Misclassification rate
        self.isRoot = False
        self.isLeaf = True              # Should be False if node has any children.
        self.depth = 0                  # Depth in the tree (distance to the head)
        self.decision = []              # Split dimension and value.
                                        # If empty, self.isLeaf should be True.
    #
#

class Tree(object):

    def __init__(self):
        self.data = []
        self.labels = []
        self.tree = []
        self.tree.append(Node())    # Initialize a head node.
        self.tree[0].isRoot = True
        self.leaves = [0]
        self.height = 1     # Number of levels from top to bottom.
        self.width = 1      # Number of leaves.

        self.unique_labels = []

        # The tree is structured as a dictionary right now:
        # tree[i] refers to node i, which is its own class defined above.
        #
        # tree[i].children:       array of pointers to children
        # tree[i].parent:         integer pointer to parent
        # tree[i].majorityLabel:  majority label.
    #

    def add_child(self,node):
        '''
        Adds a node to the tree and updates the children and
        parent of the Node objects appropriately.
        Returns the index of the new node.
        '''
        # ID of the new node
        i = len(self.tree)

        # Update the tree
        if not (self.tree[node].isLeaf):
            self.width += 1
        #
        if (self.tree[node].depth >= self.height):
            self.height += 1
        #

        # Update the parent
        self.tree[node].isLeaf = False
        self.tree.append(Node())
        self.tree[node].children.append(i)

        # Update the child
        self.tree[i].parent = node
        self.tree[i].depth = self.tree[node].depth + 1

        return i
    #

    def add_children(self,node,n=2):
        '''
        Adds n children to the specified node. Defaults to n=2.
        Returns an array of indexes pointing to the children in
        addition to adding them to self.tree.
        '''
        idxs = []
        for i in range(n):
            idxs.append( self.add_child(node) )
        #
        return idxs
    #

    def collect_leaves(self):
        '''
        Crawls down self.tree and updates self.leaves with all
        nodes for which self.tree[i].isLeaf == True. (terminal nodes)
        '''
        #self.leaves = [ (i if self.tree[i].isLeaf) for i in range(len(tree)) ]
        self.leaves = []
        for i in range(len(self.tree)):
            if self.tree[i].isLeaf:
                self.leaves.append(i)
            #
        #
        return self.leaves
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
        self.unique_labels = np.unique(self.labels)
    #

    def label_metric(self,pred_lab,true_lab,which='acc'):
        '''
        Returns accuracy (percent correctly classified)
        between predicted and true labels.
        '''
        import numpy as np

        if (which=='acc'):
            # met = float(TP+TN)/(P+N)
            met = float( np.sum(pred_lab==true_lab) )/len(true_lab)
            return met
        else:
            #print("Metrics other than accuracy are not currently supported.")
            #return None
            raise NotImplementedError("Metrics other than accuracy are not currently supported.")
        #
    #

    def misclassification_rate(self,pred_lab,true_lab):
        '''
        Returns 1-self.label_metric(pred_lab,true_lab,which='acc').
        '''
        return 1. - self.label_metric(pred_lab,true_lab,which='acc')
    #

    def choose_coordinate(self,vectors,labels):
        '''
        choosecoordinate(self,vectors,labels)

        Inputs:
            ndim-by-nvec numpy array of data vectors
            nvec numpy array of class labels, assumed in {0,1}.

        Outputs:
            Coordinate (integer) indicating the dimension on which to split.

        ----------------

        Given a list of vectors, choose the coordinate upon which to
        split based on a (normalized) heuristic within the
        sub-region:

        max_k( sum( sum( (x_ik - x_jk)**2 ) ) / sum( (x_k - mu)**2 ) )

        where the summations are done over
        elements in each of the classes; x_i are vectors with label 0,
        x_j are vectors with label 1, and the denominator is the
        variance over all vectors on the k-th coordinate.

        This should give a scale and dimension-free measure of
        how spread apart the two label sets are on that dimension,
        relative to the overall variation in that dimension.
        The intuition is that if we are locally linearly separable,
        we should see a high value, and if they are intermixed, not.

        The downside is that this is (roughly) an O(ndim*nvec**2) algorithm.

        Code is set up assuming row vectors right now.
        '''
        import numpy as np

        nvec,ndim = np.shape(vectors)

        variances = np.var(vectors,axis=0)

        # Want to exclude coordinates where there is no variation.
        excludeddims, = np.where(variances==0.)
        scores = np.zeros( ndim )


        # nl, = np.where(labels==0)
        # pl, = np.where(labels==1)
        #
        # for i in nl:
        #     for j in pl:
        #         scores += (vectors[i,:] - vectors[j,:])**2
        #     # end for
        # # end for

        for i,lab0 in enumerate( self.unique_labels[:-1] ):
            other_labs = list(self.unique_labels[:i]) + list(self.unique_labels[i+1:])
            nl, = np.where(labels==lab0)
            for j,lab1 in enumerate(other_labs):

                pl, = np.where(labels==lab1)
                scores += self.squared_set_diff(vectors,nl,pl)
                # print(lab0,lab1,scores)
            #
        #
        # if (any(variances==0.)):
        #     print(variances)
        #     print(self.data.shape)
        #     asdfasdf=input()
        # #

        scores[variances!=0.] /= variances[variances!=0.]
        scores[excludeddims] = -np.inf

        return np.argmax(scores)
    #

    def squared_set_diff(self,vectors,set0,set1):
        '''
        Given two sets of points, calculate the sum of squares
        of distances between points in set1 and set2 along axis 1.
        '''
        import numpy as np

        nvec,ndim = np.shape(vectors)
        scores = np.zeros( ndim )

        for i in set0:
            for j in set1:
                scores += (vectors[i,:] - vectors[j,:])**2
            #
        #
        return scores
    #

    def maximize_split(self,vectors,labels,k,metric='acc'):
        '''
        maximizesplit(self,vectors,labels,k,metric='bsr')

        Inputs:
            vectors: nvec-by-ndim numpy array of input data
            labels: nvec numpy array of labels, assumed {0,1}
            k: integer index indicating the dimension to split on
            metric: String indicating the metric to maximize on.

        Outputs:
            s: float; value of the split maximizing the chosen metric.

        Given a coordinate k, and a set of vectors k,
        return the value for which a chosen metric is
        maximized.

        Maximization is done using a coarse sampling for now.
        '''
        import numpy as np

        vals = vectors[:,k]
        sl,sr = vals.min(), vals.max()
        nsamps = 9 #not including the endpoints
        testsplits = (np.linspace(sl,sr,nsamps+2)[1:-1])

        mets = np.zeros( np.shape(testsplits) )
        for i,split in enumerate(testsplits):
            pred_lab = (vals < split)

            mets[i] = self.label_metric(pred_lab,labels,metric)
        #

        val = np.max( np.abs(mets-0.5) )
        maximizers, = np.where(np.abs(mets-0.5)==val)
        j = np.random.choice(maximizers)
        s = testsplits[j]

        return s
    #

    def majority_vote(self,marray):
        '''
        Returns the most common value.
        Wrapper for scipy.stats.mode.
        '''
        from scipy import stats

        mode_obj = stats.mode(marray, nan_policy='omit')

        return mode_obj.mode[0]
    #

    def split_node(self,node,vectors,labels):
        '''
        Does the actual work of creating the decision and splitting.
        I don't like that the vectors and the labels in the region need
        to be passed manually, but maybe it's okay.
        '''

        i1,i2 = self.add_children( node )

        k = self.choose_coordinate(vectors,labels)
        s = self.maximize_split(vectors,labels,k)

        self.tree[node].decision = [k,s]

        return [i1,i2]
    #

    def classify_node(self,node,true_lab):
        '''
        Does the classification based on majority vote of the input labels.
        '''

        self.tree[node].majority_label = self.majority_vote(true_lab)
        pred_lab = [ self.tree[node].majority_label for j in true_lab ]
        self.tree[node].mcr = self.misclassification_rate(pred_lab, true_lab)
        return
    #

    def recursive_node(self,node,vectors,true_lab,tol=0.):
        '''
        Does the recursion
        '''
        import numpy as np

        if (len(true_lab)==0):
            not_good = False
        else:
            self.classify_node(node,true_lab)
            not_good = (self.tree[node].mcr > tol)
        #

        if not_good:
            i1,i2 = self.split_node(node,vectors,true_lab)
            k,s = self.tree[node].decision
            mcr1 = self.tree[node].mcr

            p1, = np.where( vectors[:,k] <  s )
            p2, = np.where( vectors[:,k] >= s )

            # Main recursion
            self.recursive_node(i1,vectors[p1,:],true_lab[p1])
            self.recursive_node(i2,vectors[p2,:],true_lab[p2])
        #


        return
    #

    def grow(self,tol=0.):
        '''
        Constructs a classification tree. This version does the
        optimal splitting based on BSR, but halting the recursion
        based on a per-leaf misclassification tolerance

        (number of points misclassified)/(number of points) < tol.

        Construction is done using self.recursive_node.
        '''

        # Initialize at the head.
        node=0
        vectors = self.data
        true_lab = self.labels

        self.recursive_node(node,vectors,true_lab,tol)

        return
    #

    def classify_datum(self,datum):
        '''
        Classify a single datum by going down
        a constructed decision tree.
        '''
        node = 0
        notLeaf = not self.tree[node].isLeaf
        while notLeaf:

            idxs = self.tree[node].children
            k,s = self.tree[node].decision

            node = idxs[ datum[k] >= s ]

            notLeaf = not self.tree[node].isLeaf
        #
        return self.tree[node].majority_label
    #

    def classify(self,data):
        '''
        Based on a constructed tree, classify new data.
        '''
        import numpy as np
        npoints,ndims = np.shape(data)
        pred_labels = np.zeros(npoints)

        for i,datum in enumerate(data):
            pred_labels[i] = self.classify_datum(datum)
        #
        return pred_labels
    #

    #################
    #
    # Visualization methods.
    # As of 29 October 2017, this doesn't work. Something to do with
    # indexing of nodes in the tree.
    #

    def visualize_tree(self):
        '''
        Given an already constructed tree, visualize it.
        '''
        from matplotlib import pyplot
        import numpy as np

        def recurse_generations(treeObj,node,idxs):
            nodes = []
            coords = []
            cols = []
            adjacency = []

            j = treeObj.tree[node].depth
            # print(idxs,j)

            print(node,j,idxs[j])
            nodes = nodes + [node]
            coords.append( [idxs[j], j] )
            cols = [j]
            idxs[j]+=1

            if not treeObj.tree[node].isLeaf:
                children = treeObj.tree[node].children
                for child in children:
                    print(node,child)
                    adjacency.append([node,child])
                    # print([node,treeObj.tree[node].depth, child,treeObj.tree[child].depth])

                    nod,coo,col,con = recurse_generations(treeObj,child,idxs)

                    coords = np.concatenate( (coords,coo) )
                    cols = np.concatenate( (cols,col) )
                    nodes = nodes + nod
                    if len(con)>0:
                        adjacency = adjacency + con
                    #
                #
            #

            return nodes,coords,cols,adjacency
        #

        fig,ax = pyplot.subplots(1,1)
        colors = pyplot.cm.rainbow(np.linspace(0,1,self.height))

        idxs = np.zeros(self.height+1)
        nodes,coords,cols,adj = recurse_generations(self,0,idxs)
        print(coords)
        coords = np.random.randn(np.shape(coords)[0], np.shape(coords)[1])

        ax.scatter(coords[:,0],coords[:,1],c=cols, s=100)
        for edge in adj:

            x0,y0 = coords[edge[nodes[0]]][0], coords[edge[nodes[0]]][1]
            dx,dy = coords[edge[nodes[1]]][0]-x0, coords[edge[nodes[1]]][1]-y0
            ax.arrow(x0,y0,dx,dy, color='k', length_includes_head=True,width=0.01,head_width=0.1,head_length=0.2,overhang=0.1)
        #
        # print(nodes)
        for i,node in enumerate(nodes):
            print(coords[i][0],coords[i][1], "%i"%node)
            ax.text(coords[i][0]+0.2, coords[i][1]+0.2, "%i"%node, fontsize=14,ha='center',va='center')
        #

        return fig,ax,coords,cols,adj
    #
#
