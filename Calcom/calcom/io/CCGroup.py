from .CCList import CCList

class CCGroup():

    def __init__(self, converter=None):
        # The names of the datasets (GSE40012, GSE20346, etc.)
        self.dataset_names = CCList()

        # A dictionary mapping the above strings to their CCDataSet
        self.datasets = {}

        # A CCConverter converter object. If provided, it will convert datasets
        # before being added. Since added datasets are passed by reference, it
        # mutates the passed dataset passed. If None or not provided, then
        # performs no conversion.
        self.converter = converter

        # A master list of feature names; this will be the union (numpy.union1d
        # of ccd.variable_names across all CCDataSets). Since numpy.union1d is
        # used, *this is always sorted*
        self.variable_names = CCList()

        # A dictionary mapping each value ("GSE40012", etc) to a list of
        # pointers for the master list of feature names. These pointers show
        # which variableNames the dataset actually uses.
        #
        # I.e., self.variableNameLocations[datasetName][3] gives you the index
        # of the fourth variableName in the master list. The fourth
        # variableName is determined by the sorted order of the variable list
        # of that dataset.
        self.variableNameLocations = {}

        # A dictionary for dataset name to a dictionary of variable names to
        # index in dataset's variable_names. It's private because it's
        # expensive to computer and not very useful outside of the class.
        self._variableNameLocationsInverted = {}
    #

    def genName(self, base="Data Set "):
        """Returns a generated name for the dataset"""
        i = len(self.dataset_names)
        while (base+str(i)) in self.dataset_names:
            i+=1
        return (base+str(i))
    #

    def populateVariableNameLocationsInverted(self, name):
        self._variableNameLocationsInverted[name] = dict(
            [(e,i) for i,e in enumerate(self.datasets[name].variable_names)]
        )
    #

    def populateVariableNameLocations(self):
        """Populates variableNameLocations

        Assumptions: self.variable_names contain no duplicates and inverted
        variable is set. inverted is set when dataset is added

        This function is efficient and passes the basic test
        """
        self.variableNameLocations = {}
        for name in self.dataset_names:
            # My probaby wrong guess is that this is O(n log n)
            # Prepopulate array for random access
            self.variableNameLocations[name] = [None]*len(self.datasets[name].variable_names)
            # for every variable_name
            for loc,vn in enumerate(self.variable_names): # n
                # if it applies to the dataset
                if vn in self._variableNameLocationsInverted[name]: # log n
                    # add its index to variableNameLocations
                    ds_loc = self._variableNameLocationsInverted[name][vn] # log n
                    self.variableNameLocations[name][ds_loc] = loc # 1
        #

        # We want to remove the None
        # entries from variableNameLocations. This might be slow.
        for name in self.dataset_names:
            poplocs = []
            for i,elem in enumerate(self.variableNameLocations[name]):
                if elem==None: poplocs.append(i)
            #
            for i in poplocs[::-1]:
                self.variableNameLocations[name].pop(i)
            #
        #
        return
    #

    def getIntersection(self, datasets=None):
        """
        Returns the intersection of variableNameLocations[datasets][:] sorted
        by their index in variable_names
        """
        import numpy as np

        intersection = []
        if not datasets:
            datasets = self.dataset_names
        for key in datasets:
            if len(intersection)>0:
                intersection = np.intersect1d(
                        intersection,
                        self.variableNameLocations[key]
                )
            else:
                intersection = self.variableNameLocations[key]
        intersection.sort()
        return intersection
    #

    def addCCDataSet(self, ccd, name=None):
        """
        A function taking a list of CCDataSets and optionally a list of dataset
        names and populating the things above.
        """
        if not name:
            name = self.genName()
        elif name in self.dataset_names:
            name = self.genName(name)
        self.dataset_names.append(name)
        self.datasets[name] = ccd
        if self.converter:
            ccd = self.converter.convert(ccd)
        # self.variable_names = np.union1d(self.variable_names, ccd.variable_names)
        set_variable_names = set(self.variable_names) # change to log n search time
        for vn in ccd.variable_names:
            if vn not in set_variable_names:
                self.variable_names.append(vn)
                set_variable_names.add(vn)
        self.populateVariableNameLocationsInverted(name)
        self.populateVariableNameLocations()
    #

    def addCCDataSets(self, ccds, ccd_names=[]):
        '''
        Calls addCCDataSet for each entry in the lists
        ccds and ccd_names.
        '''
        for i in range(len(ccds)):
            if i < len(ccd_names):
                name = ccd_names[i]
            else:
                name = None
            self.addCCDataSet(ccds[i], name)
        #

        return
    #

    def createCCDataSet(self,**kwargs):
        """
        A function taking in two values corresponding to two CCDataSets output
        a new CCDataSet where:
            * Only variables shared between the two are kept; the rest are
              thrown out. This involves some intersections using the pointer
              dict above
            * Datapoints from each are merged into new one otherwise, without
              any regard to cimilarities/differences
            * (New attribute/added based on strings above)??? cut off

        Optional inputs:
            datasets: A list of strings indicating which datasets to work on.
                Default: self.dataset_names (all datasets in the group)
            verbosity: A positive value prints any warnings or errors.
                Default: 0
            import_feature_sets: Boolean. If True, feature sets from each of
                the datasets are examined. If they lie in the the set of new
                features, the indexing for that feature set is recreated.
                Partial feature matches are kept and renamed appropriately.
                Feature sets with duplicate names are handled by appending an
                integer to the name (e.g., 'fset_0', 'fset_1')
            fillData: switch to control whether to delete non-intersection
                variables or leave them as np.nan (or something) to be imputed
                later.
        """
        import numpy as np
        import calcom
        from calcom.io import CCList

        datasets = kwargs.get('datasets',self.dataset_names)
        verbosity = kwargs.get('verbosity',0)
        import_feature_sets = kwargs.get('import_feature_sets',True)
        # fillData = kwargs.get('fillData',None)

        intersection = self.getIntersection(datasets)
        ccd = calcom.io.CCDataSet()

        if import_feature_sets:
            for name in datasets:
                for featList in self.datasets[name].feature_sets:
                    srcIndexes = self.datasets[name].feature_sets[featList] # reference, not copy
                    destIndexes = []
                    for index in srcIndexes:
                        destIndex = self.variableNameLocations[name][index]
                        # if all datasets have that feature or if a value to
                        # fill missing values with is specified, add the
                        # feature.
                        # if destIndex in intersection or fillData:
                        loc = np.where(np.array(intersection)==destIndex)[0]
                        if len(loc)>0:
                            destIndexes.append(loc[0])
                    #

                    # get feature list's name
                    if featList in ccd.feature_sets:
                        i = 0
                        while (featList+'_'+str(i)) in ccd.feature_sets:
                            i+=1
                        featName = featList+str(i)
                    else:
                        featName = featList

                    ccd.add_feature_set(featName, destIndexes)
        #

        attrnames = CCList()
        attrdescs = {}
        attrvalues = CCList()
        variable_names = CCList()
        datamat = CCList()

        title_key = 'source_dataset'
        attrnames = [title_key]
        attrdescs[title_key] = 'The dataset this sample belongs to'

        # populate variables
        # set attrnames and attrdescs
        for name in datasets:
            attrnames = np.union1d(attrnames, self.datasets[name].attrnames)
            for attrname in self.datasets[name].attrnames:
                attrdescs[attrname] = self.datasets[name].attrs[attrname].long_description
                # Note: this could overwrite descriptions, but they _should_ be
                # the same.

        # populate attrvalues
        # it should be the transpose of attribute name by values. The values is
        # a list corresponding to the order of the subjects
        attrvalues = {}
        for name in datasets:
            length = len(self.datasets[name].data)
            for attrname in attrnames:
                if attrname not in attrvalues:
                    attrvalues[attrname] = CCList()
                if attrname == title_key:
                    attrvalues[attrname].extend([name for _ in range(length)])
                elif attrname == "attrnames":
                    continue
                elif attrname in self.datasets[name].attrnames:
                    attrvalues[attrname].extend(self.datasets[name].get_attrs(attrname))
                else:
                    attrvalues[attrname].extend([None for _ in range(length)])
        attrvalues = np.array([attrvalues[attrname] for attrname in attrnames]).transpose()
        attrdescs = CCList([attrdescs[attr] for attr in attrnames])

        variable_names = CCList([self.variable_names[i] for i in intersection])
        variable_names_set = set(variable_names)
        for name in datasets:
            toAppendIndexes = []
            set_variable_names = set(self.datasets[name].variable_names)
            for vn in variable_names:
                # inefficient for the same reason as populateVariableNameLocations...
                if vn in set_variable_names:
                    # get index of vn in its variable_names
                    #index = self.datasets[name].variable_names.index(vn)
                    index = self._variableNameLocationsInverted[name][vn] # from n to log n
                    toAppendIndexes.append(index)
                else:
                    toAppendIndexes.append(-1)

            for subj in self.datasets[name].data:
                # toAppend = [(subj[i] if i != -1 else fillData) for i in toAppendIndexes]
                toAppend = [(subj[i] if i != -1 else None) for i in toAppendIndexes]
                datamat.append(toAppend)

        ccd.add_attrs(attrnames,attrdescs)
        ccd.add_datapoints(datamat,attrnames,attrvalues)
        ccd.add_variable_names(variable_names)

        return ccd
#

if __name__ == '__main__':
    import calcom

    import os
    from pathlib import Path

    def loadData(filename):
        datafile = Path(filename)
        try:
            my_abs_path = datafile.resolve()
        except FileNotFoundError:
            return None
        else:
            ccd = calcom.io.CCDataSet()
            ccd.load_from_disk(filename)
            return ccd

    ccd1 = loadData('ccd_gse20346.h5')
    if not ccd1:
        import load_GSE20346_series_matrix
        ccd1 = load_GSE20346_series_matrix.ccd

    ccd2 = loadData('ccd_gse40012.h5')
    if not ccd2:
        import load_GSE40012_series_matrix
        ccd2 = load_GSE40012_series_matrix.ccd

    print('datasets loaded')
    ccg = CCGroup()
    print('adding first dataset')
    ccg.addCCDataSet(ccd1, "GSE20346")
    print('adding second dataset')
    ccg.addCCDataSet(ccd2, "GSE40012")
    print('creating dataset')
    ccd = ccg.createCCDataSet()
    print(ccd)
