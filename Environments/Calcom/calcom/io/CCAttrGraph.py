
class CCAttrGraphNode:
    '''
    A single node in the graph. Just holds some basic information.
    Most critically, self.name should be the same as the attribute name
    in
    '''
    def __init__(self,*args):
        self.name = None
        self.parents = []
        self.children = []
    #
#

class CCAttrGraph(dict):
    '''
    A class which encodes a directed graph structure of CCDataAttr(s).
    The purpose is to explore how weakenings of labels (where possible)
    affects classification rates. For example:
        - If labels are multiple strains of mice and ferrets,
          how do classification rates change if we look at the relaxed
          problem using only all mice and all ferrets?
        - If labels are a set of time points, how do classification rates
          change if we look at the relaxed problem, allowing the time
          to be correct within a window? (This might feel similar to a
          discrete version of a continuum regression problem.)

    The idea here is that we can relax the labels for free, without
    needing to fit new models for each new classification problem.
    '''

    def __init__(self,*args):
        dict.__init__(self,args)
        self.nodes = []
        self.edges = []
    #

    def create_nodes(self,attrs):
        '''
        Initialize the node/vertex set of attributes, without
        specifying the edges. This can be done for "free"
        given the attrs in a CCDataSet.
        '''
        for attr in attrs:
            node = CCAttrGraphNode()
            node.name = attr.name
            node.attr = attr
            self.nodes += [node]
        #
        return
    #

    def grow_edges_numeric(self,attrname,radius=1,overlap=True):
        '''
        Given an attribute which is numeric, generate the edges
        using a given radius (in the units of that attribute).

        If overlap=False, non-overlapping intervals of time are used;
        this would be preferred if for example you wanted to group
        all time points between 0 and 24 into day 1,
        hours 24 to 48 in day 2, etc. Otherwise (by default), edges are drawn
        using true intervals centered around each label.
        '''
        return
    #
#
