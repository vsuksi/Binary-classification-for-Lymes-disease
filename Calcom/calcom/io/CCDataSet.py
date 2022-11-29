'''
The purpose of the CCDataSet is to generalize the
importing of a dataset from an experiment with multiple metadata and/or
attributes attached to each sample point. For very regularly structured
data, a simple table for the metadata with corresponding pointers to
the data suffices just as well, but we tend to frequently encouter data
which has irregular structure to it, for which much additional information
needs to be generated and processed.

Additionally, handling large datasets result in extremely slow loading times
for dealing with, e.g., csv files, and loading/saving to a binary format
(we opt for HDF) is automated through functions in this class.

The simplest example to populate an instance of this class is:

    import calcom
    ccd = calcom.io.CCDataSet()
    ccd.create(data,metadata)

where "data" is a (possibly irregularly shaped) list of datapoints,
and metadata is a regularly shaped array whose first row is a list of
names for the attributes (each piece of metadata), and following rows
give corresponding values for each datapoint.

'''

class CCDataSet:
    '''
    The big data container. Keeps all the CCDataPoint(s), populates them,
    keeps track of the attributes, etc.
    '''
    def __init__(self, fname="", preload=True, print_about=True):
        from calcom.io import CCList,CCDataAttr

        self.version = "1.1"    # Should be changed everytime the layout(python attributes) of the class(or the subclasses);
        self.data = CCList()
        self.attrnames = CCList() # Names of attributes that can be called using getattr()
        self.attrs = dict()     # Dictionary of CCDataAttr's used to populate data.
        self.ids = CCList()           # List of used IDs
        self.feature_sets = dict()    # Dictionary of lists indicating set of features/attributes
        # self.attrgraph = None

        idattr = CCDataAttr()
        idattr.name = '_id'
        idattr.long_description = "CCDataSet ID. These are unique identifiers for each CCDataPoint."
        self.attrs[idattr.name] = idattr
        self.variable_names = CCList()
        self.variable_descrs = CCList()
        self.study_attrs = dict()
        self.fname = None

        self._preloaded = False
        self._metadata_table = None

        self._about_str = ''    # Raw string describing the dataset. Can either be modified
                                # directly or through the function self.add_about().
                                # Will be printed with self.about().

        # THIS MUST ALWAYS BE THE END OF THE INIT!
        if fname!="":
            # Note self.load() will print the "about" string after successful load.
            self.load(fname, preload=preload, print_about=print_about)
            self._preloaded = preload   # This is reached if load_from_disk doesn't break
        #
        
        return

    #

    def _generate_id(self,d=10,**kwargs):
        '''
        Generates a d-digit string id number for a new entry in the
        dataset.  Optionally, a prefix and/or suffix can be specified.
        Checks against everything in self.ids to ensure there are no
        overlaps.

        If a prefix or suffix is specified, they are appended to the
        d-digit number.

        For example:
            id = _generate_id(4,prefix='H_',suffix='_ok')
            print(id)

        would return something of the form "H_0123_ok". Both prefix and
        suffix default to "" (empty string).

        (IN DISTANT FUTURE: efficient search of self.ids? Tree structured
        by digits?)
        '''
        import numpy as np

        prefix = str( kwargs.get('prefix',"") )
        suffix = str( kwargs.get('suffix',"") )

        # Hard-coded ratio of number of times to attempt to find an id
        # before an error message is printed.  This is in log10 units, so
        # -1 corresponds to 1/10.
        errratio = -1

        idd = np.random.randint(0,10**d, dtype=np.int64)
        nattempts = 1
        while (idd in self.ids):
            idd = np.random.randint(0,10**d)
            nattempts += 1
            if (nattempts == 10**(d+erratio)):
                raise ValueError("It is taking too many attempts to generate a unique ID.\nYou need to use more digits for the current dataset.")
            #n
        #
        idd = prefix + str(idd).zfill(d) + suffix
        return idd
    #

    def generate_str_ids(self,**kwargs):
        '''
        Purpose: to automatically generate human-readable unique
        identifiers for each datapoint.

        Inputs: None
        Outputs: list of string identifiers, unique to each datapoint

        Optional inputs:
            apply: Boolean. If True, the existing ccd.data[i]._id is
                replaced for all data. Default: False
            attrs: List-like of one or more attributes to prioritize
                for automatic identifier generation. See below for details.

        Implementation: happens in two passes:
            1. A ranking of ccd.attrnames is created based on the
                number of unique values for each attribute, in descending
                order. If the user specifies any attributes by the
                optional argument, those are automatically placed in the
                front of the ordering.
            2. Inside a loop over this ordering, strings for each datapoint
                are appended with the corresponding attribute values.
                The number of unique identifiers are evaluated, and the
                loop breaks if it is the same as the number of datapoints.
            3. If this cannot be done (data not uniquely identified by the
                collection of all its metadata) then the existing strings
                are suffixed with zero-padded integers placed
                according to the ordering of the existing strings.
        '''
        # attrs = ['SubjectID', 'StudyID']
        import numpy as np

        attrs_vip = kwargs.get('attrs', [])

        n = len(self.data)

        n_uniques = np.array([ len(np.unique(self.get_attrs(an))) for an in self.attrnames ])
        n_unique_ranking = np.argsort(-n_uniques)
        attrs_ordered = list( self.attrnames )
        attrs_ordered = np.array(attrs_ordered)[n_unique_ranking]
        attrs_ordered = list( attrs_ordered )

        # Send the VIP attributes to the front of the list.
        for an in attrs_vip:
            attrs_ordered.remove(an)
        attrs_ordered = list(attrs_vip) + attrs_ordered

        candidate_identifiers = ['' for _ in range(n)]
        n_unique = 1  # how many unique strings are there?

        idx = 0
        while ( (n_unique < n) and (idx < len(attrs_ordered)) ):
            sep = ['','_'][idx>0]
            attr = attrs_ordered[idx]
            aval = self.get_attrs(attr)
            candidate_identifiers = [c + sep + str(aval[i]) for i,c in enumerate(candidate_identifiers)]
            n_unique = len(np.unique(candidate_identifiers))
            idx += 1
        #
        if (n_unique < n):
            # nuclear option: zero-padded integers.
            sep = ['','_'][idx>0]
            oom = int(np.log10(n)) + 1  # order of magnitude.
            aval = [ str(i).zfill(oom) for i in range(n) ]
            candidate_identifiers = [c + sep + str(aval[i]) for i,c in enumerate(candidate_identifiers)]
        #

        if kwargs.get('apply', False):
            for i in range(n):
                self.data[i]._id = candidate_identifiers[i]
        #

        return candidate_identifiers
    #

    def add_about(self,description='',**kwargs):
        '''
        Replaces self._about_str with the input of this function.

        Inputs:
            description: A raw string (newlines, tabs, etc, are okay)
        Outputs:
            none.
        Optional inputs:
            save_to_disk: whether to immediately save the description to the
                dataset on file (default: False). Note this function will
                fail if self.fname isn't defined.
            add_autosummary: whether to append an automatic summary of the
                structure of the data and attributes to the bottom
                of the description using self.autosummarize(). (Default: True)
        '''

        full_descr = str(description)
        if kwargs.get('add_autosummary',True):
            autostr = self.autosummarize(return_str=True)
            separator = '\n|\n|\n|' + '='*60 + '\n|'*2 + '\n\n'
            full_descr = full_descr + separator + autostr
        #

        self._about_str = full_descr

        if kwargs.get('save_to_disk',False):
            # This will fail if the dataset hasn't been saved yet.
            import h5py
            h5f = h5py.File(self.fname)
            h5f.attrs['about'] = self._about_str
            h5f.close()
        #
        return
    #

    def about(self):
        '''
        Prints self._about_str. That's it.

        Inputs: none
        Outputs: none
        '''
        print(self._about_str)
        return
    #

    def readme(self):
        '''
        Alias for self.about().
        '''
        self.about()
        return
    #

    def autosummarize(self, list_up_to=4, max_str_len=60, return_str = False):
        '''
        Looks over all attributes in the datasets and
        prints a nice summary of them. For each attribute,
        a small number of attribute values are listed as well.

        Inputs: None.
        Optional inputs:
            list_up_to: integer; maximum number of example unique attribute
                values are listed; sorted by number of occurrences. (default: 3)
            max_str_len: integer; maximum length of strings printed
                (enforced mainly for attribute values/names).
            return_str: boolean; whether to return a raw string corresponding
                to the usual output (default: False).
        Outputs:
            None, if return_str==False; else the raw string of the summary.

        Notes: This can be useful on its own. This is also used as an argument
        to self.add_about(), which puts the summary on the bottom of the
        manually generated description.
        '''

        # TODO: ADD SUPPORT FOR kwarg "idx"

        import numpy as np
        msl = max_str_len
        ellipses = '...'
        def tr(instr):
            l = max_str_len-len(ellipses)
            if len(instr) > l:
                return instr[:l] + ellipses
            else:
                return instr
        #

        study_attrs = list(self.study_attrs)
        attrs = list(self.attrs)
        attrs.remove('_id')
        try:
            attrs.remove('id')  #datasets have this floating around even though they shouldn't.
        except:
            pass
        #

        header = 'What follows is an automatically generated summary of the dataset.\n'
        study_attr_str = 'There are %i study attributes:\n'%len(study_attrs)
        for s in study_attrs:
            name = self.study_attrs[s].name
            val = self.study_attrs[s].value
            study_attr_str += '\t%s : %s\n' % (tr(name),tr(val))
        #

        data_str = 'There are %i CCDataPoints in the dataset.\n'%len(self.data)

        # Check the shapes of each datapoint.
        l0_shape = [np.shape(d) for d in self.data]
        is_l0_shape_homogeneous = np.all([l0i==l0_shape[0] for l0i in l0_shape])
        insert = 'homogeneous' if is_l0_shape_homogeneous else 'heterogeneous'

        data_str += 'Data points have ' + insert + ' shape'
        if is_l0_shape_homogeneous:
            data_str += ' with shape %s.\n'%str(l0_shape[0])
        else:
            data_str += '; first few shapes are:\n'
            ub = min(list_up_to,len(self.data))
            for i in range(ub):
                data_str += '\t' + str(l0_shape[i]) + '\n'
            #
            if len(self.data)>=i:
                data_str += '\t' + ellipses + '\n'
        #

        # Check the dtypes of each datapoint; generally expect
        # to split between heterogeneous data (np.dtype('O')) or
        # nice regular arrays which are np dtypes of int, float, etc.
        l0_type = [d.dtype for d in self.data]
        is_l0_type_homogeneous = np.all([l0ti==l0_type[0] for l0ti in l0_type])
        if is_l0_type_homogeneous and l0_type[0]==np.dtype('O'):
            # Expect heterogeneous elements; investigate if
            # these elements have the same structure across all datapoints.
            l1_shape = [[np.shape(e) for e in d] for d in self.data]
            is_l1_shape_homogeneous = np.all([l1s==l1_shape[0] for l1s in l1_shape ])
            if not is_l1_shape_homogeneous:
                data_str += 'Heterogeneous data within datapoints; the first few are:\n'
                ub = min(list_up_to,len(self.data))
                for i in range(ub):
                    if i<ub-1:
                        data_str += str(l1_shape[i]) + ',\n'
                    else:
                        data_str += str(l1_shape[i]) + '\n'
                #
        #

        attr_str = 'There are %i attributes:\n'%len(attrs)
        all_eqs = np.array( [self.partition(a) for a in attrs] )
        # Get an ordering of attributes based on number of unique
        # values attained. Print them according to this.
        # all_nuniques = np.array( [len(e[0]) for e in all_eqs] )
        all_nuniques = np.array([len(e.keys()) for e in all_eqs])

        order = np.argsort( all_nuniques )
        for o in order:
            a = attrs[o]
            # u,eq = all_eqs[o]
            eq = all_eqs[o]
            u = list(eq.keys())
            isplural = (len(u)>1)
            if isplural:
                attr_str_single = '\t%s (%i unique values):\n'%(a, len(u))
            else:
                attr_str_single = '\t%s (%i unique value):\n'%(a, len(u))
            #

            # List attribute values in decreasing order.
            a_order = np.argsort( [ -len(eq[uv]) for uv in u ] )
            for i,a_o in enumerate(a_order[:min(list_up_to,len(u))]):
                attr_str_single += '\t\t%s (%i entries)\n'%( u[a_o], len(eq[u[a_o]]) )
            #
            # If there are values we haven't printed, add a downward ellipses.
            if i==list_up_to-1 and len(u)>=i:
                attr_str_single += '\t\t%s\n'%ellipses
            #

            attr_str += attr_str_single + '\n'
        #

        full_str = header + '\n' + data_str + '\n' + attr_str + '\n' + study_attr_str

        if return_str:
            return full_str
        else:
            print(full_str)
            return
        #
    #

    def add_feature_set(self, feature_set_name, features, **kwargs):
        '''
        Define a new subset of features based on _pointers_ to variables
        in the current dataset. That is, a feature_set of [0,5,12] will
        correspond to a feature set with names

            [ccd.variable_names[k] for k in [0,5,12]].

        If you want to add a feature set based on variable names, see
        self.add_variable_set() which instead takes a
        list of strings and compares against ccd.variable_names before
        passing the maximal number of matching pointers to this function.

        Input:
            feature_set_name: String indicating the name of the feature
                set.
            features: List of integer pointers.

        Outputs: None.

        Optional inputs:
            save_to_disk: Boolean; whether to directly save the feature set
                to the HDF file on disk. (default: False)
        '''
        from calcom.utils import type_functions as tf
        from calcom.io import CCList

        save_to_disk = kwargs.get('save_to_disk', False)


        if type(feature_set_name)!=str:
            raise ValueError("Expected a string")

        # This is less strict but might get us in trouble at some point.
        list_like = tf.is_list_like(features)

        if list_like and all(tf.is_integer(item) for item in features):
            self.feature_sets[feature_set_name] = CCList(features)
            if save_to_disk and self.fname:
                from calcom.io import save_feature_set
                save_feature_set(self.fname,feature_set_name,features,**kwargs)
        else:
            raise ValueError("Expected a list of integers")

        return
    #

    def add_variable_set(self, feature_set_name, variable_set, **kwargs):
        '''
        Define a new subset of features based on _strings_ corresponding
        to variables in the current dataset. That is, a variable_set of
        ['gene1', 'gene121', 'gene155'] will attempt to match with
        variable names in ccd.variable_names by (roughly speaking) the
        comparison

            [numpy.where(v==numpy.array(ccd.variable_names))[0][0] for v in variable_set]

        where empty results in numpy.where are ignored. The result of
        this comparison is passed to self.add_feature_set().

        If you want to add a feature set based on pointers, see
        self.add_feature_set().

        Input:
            feature_set_name: String indicating the name of the feature set.
            variable_set: List of strings of variable names.

        Outputs: None.

        Optional inputs:
            save_to_disk: Boolean; whether to directly save the feature set
                to the HDF file on disk. (default: False)
            verbosity: Indicating level of output. (default: 0)
        '''
        import numpy as np
        from calcom.utils import type_functions as tf
        import numpy as np


        save_to_disk = kwargs.get('save_to_disk', False)
        verb = kwargs.get('verbosity', 0)


        if type(feature_set_name)!=str:
            raise ValueError("Expected a string")

        # if isinstance(feature_set,list) and all(isinstance(item, int) for item in feature_set):

        # This is less strict but might get us in trouble at some point.
        list_like = tf.is_list_like(variable_set)

        if verb>0: print('Attempting to add a variable set with %i elements...'%len(variable_set))
        if list_like and all(tf.is_string_like(item) for item in variable_set):
            ptrs = []
            # TODO: optimize this algorithm.
            ref_vnames = np.array(self.variable_names)
            for v in variable_set:
                locs = np.where(v==ref_vnames)[0]
                if len(locs)==0:
                    if verb>1: print('No match found for variable %s in the CCDataSet.'%v)
                    continue
                else:
                    if verb>1: print('Variable %s matched with self.variable_names[%i]'%(v,locs[0]))
                    ptrs.append(locs[0])
            #
        else:
            raise ValueError("Expected a list of integers")
        #
        if verb>1: print('%i of %i variables matched for the feature set.'%(len(ptrs),len(variable_set)))
        if verb>0: print('done.')

        # Take the pointers and pass them on.
        self.add_feature_set(feature_set_name, ptrs, **kwargs)

        return
    #

    #######################################################################

    def add_attrs(self,attrnames,attrdescrs=None,**kwargs):
        '''
        Initializes a new CCDataAttr for each element in attrnames.
        '''
        if kwargs.get('show_deprecation_warning',False):
            import warnings
            warnings.simplefilter('always')
            warnings.warn('This function will soon be moved to be private;'+
                ' it is recommended to use self.create() for dataset creation instead.',PendingDeprecationWarning)
            warnings.simplefilter('default')
        #

        from calcom.io import CCDataAttr,CCList

        if type(attrnames)==str:
            newattr = CCDataAttr()
            newattr.name = attrnames
            newattr.long_description = attrdescrs
            # self.attrs += [newattr]
            self.attrs[attrnames] = newattr
            self.attrnames.append(attrnames)
        else:
            # Assumed list/iterable input.
            for i,attrname in enumerate(attrnames):
                newattr = CCDataAttr()
                newattr.name = attrname
                if type(attrdescrs)==type(None):
                    newattr.long_description = attrdescrs
                else:
                    newattr.long_description = attrdescrs[i]
                #
                # self.attrs += [newattr]
                self.attrs[attrname] = newattr
            #
            
            # cast to list - pop/remove functionality needed elsewhere
            self.attrnames = CCList( attrnames )
        #
        return
    #

    def append_attr(self,attrname,attrvals,attrdescr=None,**kwargs):
        '''
        Define a new attribute on the datapoints in the dataset
        with a list of values; one for each datapoint.  The type for the
        new attribute is inferred from the first entry in attrvals.

        Inputs:
            attrname: String, name of new attribute
            attrvals: list or numpy.ndarray of values for the attribute for
                each datapoint.

            attrdescr: String, long description of new attribute (default:
                None)

        Optional inputs:
            is_derived: Boolean, whether the variable is derived from the
                original data (defalt: True)
            idx: list of integers. Pointers to the subset of data on which
                to apply this function. (default: all data)

        '''
        import copy
        import numpy as np

        from calcom.utils import type_functions as tf
        from calcom.io import CCList


        if not tf.is_list_like(attrvals):
            raise ValueError("attrvals must be list like")

        if not tf.is_string_like(attrname):
            raise ValueError("attrnames must be string like")

        if ('idx_list' in kwargs) and ('idx' not in kwargs):
            # just keep it for backward compatibility; don't advertise it.
            idx = kwargs.get('idx_list', np.arange(0,len(self.data), dtype=np.int64))
        else:
            idx = kwargs.get('idx', np.arange(0,len(self.data), dtype=np.int64))
        #

        if len(idx) != len(attrvals):
            raise ValueError("Number of attribute values does not match the number of data points")

        self.add_attrs([attrname],[attrdescr])

        self.attrs[attrname].is_derived = kwargs.get('is_derived',True)
        self.attrs[attrname].is_numeric = tf.is_numeric(attrvals[0])
        self.attrs[attrname].type = type(attrvals[0])

        for i, k in enumerate(idx):
            datum = self.data[k]
            mo = copy.deepcopy(self.attrs[attrname])    # Wish there were a cleaner way.
            datum.set_attrs([mo], [attrvals[i]])

        # NOTE: It's late and I can't find the bug. Just refresh the names.
        self.attrnames = CCList(self.attrs.keys())
        return
    #

    def delete_attr(self,attrname,**kwargs):
        '''
        Deletes an attribute from the list and all data points.

        Inputs:
            attrname: string, name of the attribute.
        Optional inputs:
            show_warning: Boolean. One-time check to confirm deletion of
                all the study-wide attributes from datapoints.  (default:
                True)
        Outputs:
            None
        '''
        show_warning = kwargs.get('show_warning',False)

        if show_warning:
            print('\n\tWARNING: Attribute %s will be deleted from the dataset.\n\n\ty to continue; anything else to cancel.\n'%attrname)
            # print('Attribute %s will be deleted from the dataset. Type y to continue')
            cont = input()
            if cont != 'y':
                return
            #
        #
        for datum in self.data:
            delattr(datum,attrname)
            datum.attrnames.remove(attrname)
        #
        self.attrs.pop(attrname)
        self.attrnames.remove(attrname)

        return
    #

    def delete_all_attrs_except(self,attrlist,**kwargs):
        '''
        Deletes all EXCEPT a given list of attribute names.

        Inputs:
            attrlist: a list of strings with names of the attributes to
                keep.
        Optional inputs:
            show_warning: Boolean. One-time confirmation of deletion,
                showing a list of names of everything that will be deleted.
                (default: True)
            verbosity: integer. If greater than 0, information about the
                attributes is printed during the loop. (default: 1)
        Outputs:
            None
        '''
        import warnings
        warnings.simplefilter('always')
        warnings.warn('This function exists only for backward compatibility; pending deletion.',PendingDeprecationWarning)
        warnings.simplefilter('default')

        from calcom.io import CCList

        show_warning = kwargs.get('show_warning',True)
        verbosity = kwargs.get('verbosity',1)

        # Filter out the stuff we don't want removed.
        target_attrs = CCList(self.attrnames)
        for attrname in attrlist:
            try:
                target_attrs.remove(attrname)
            except:
                continue
            #
        #

        # Show a list of everything that's going to be deleted.
        if show_warning:
            print('\n\tWARNING: The following list of attributes will be deleted from the dataset:\n')
            for attrname in target_attrs:
                print('\t\t'+attrname)
            #
            print('\n\n\ty to continue; anything else to cancel.\n')
            cont = input()
            if cont != 'y':
                return
            #
        #

        for attrname in target_attrs:
            self.delete_attr(attrname,show_warning=False)
        #
        return
    #

    def cleanup_attrs(self,**kwargs):
        '''
        If any attribute which has the same value amongst all entries, then
        a new top-level attribute called self.study_attrs is created which
        contains a single copy of each of these. Then the attributes are
        deleted from the data and self.attrs.

        Input:
            None
        Optional inputs:
            show_warning: Boolean. One-time check to confirm deletion of
                all the study-wide attributes from datapoints. (default:
                False)
            verbosity: integer. If greater than 0, information about the
                attributes is printed during the loop. (default: 0)
            keep: list-like. Any attribute names in this list are ignored
                during the clean-up. The id is always kept, even if this is
                specified. (default: ['_id'])
        Output:
            None
        '''
        # import warnings
        # warnings.simplefilter('always')
        # warnings.warn('This function is deprecated and will soon be deleted.', DeprecationWarning)
        # warnings.simplefilter('default')

        import numpy as np
        from calcom.io import CCList

        show_warning = kwargs.get('show_warning',False)
        verbosity = kwargs.get('verbosity',0)
        keepers = CCList( kwargs.get('keep',['_id']) )
        
        if '_id' not in keepers:
            keepers = keepers.append( '_id' )

        if show_warning:
            print('\n\tWARNING: Any study-wide attributes will be removed from datapoints,\n\tand moved to self.study_attrs.\n\n\ty to continue; anything else to cancel.\n')
            cont = input()
            if cont != 'y':
                return
            #
        #

        if 'study_attrs' not in list(self.__dict__.keys()):
            self.study_attrs = {}
        #
        for attrname in self.attrnames[::-1]:

            if attrname not in keepers:
                numuniques = len(np.unique(self.get_attrs(attrname)))
                if verbosity>1:
                    print('Attribute %s has %i unique values.'%(attrname,numuniques))
                #
                if numuniques<=1:
                    attr = getattr(self.data[0],attrname)
                    self.study_attrs[attrname] = attr
                    self.delete_attr(attrname,silent=True)
                    if verbosity>0:
                        print('Attribute %s has been moved to self.study_attrs.'%attrname)
        #

        return
    #

    def add_datapoints(self,dataarray,newattrs,attrvals,**kwargs):
        '''
        Args:
            dataarray: a 2d numpy array subject by sample
            newattrs: a 1d list of string attribute names
                ['attrname1', 'attrname2', ...]
                note: len(dataarray) == len(attrvals)
            attrvals: np.array of np.arrays of attributes for subject
                [[atr1, atr2, ...], ...]
                note: len(newattrs) == len(attrvals[0])

        Takes a data array (2d numpy array) and creates a new entries in
        self.data for each row in the numpy array, using all the attributes
        in self.attrs.

        Currently, you need to define all the attributes prior to calling
        this!  Unless stated otherwise, data points won't update if new
        attributes are added later.

        IF YOU HAVE ARRAYS AS ATTRIBUTES YOU *MUST* INPUT THE ATTRVALS AS A
        NUMPY ARRAY USING 'OBJECT' DTYPE. FOR EXAMPLE:

        WON'T WORK:
            attrvals = ['p','q',-20, [1,2,3]]
            attrvals = [attrvals]
        WILL WORK:
            attrvals = ['p','q',-20, [1,2,3]]
            attrvals = [ np.array(attrvals, dtype='O') ]

        NOTE THAT DATAARRAY AND ATTRVALS ARE EXPECTED AS ITERABLES. IF YOU ARE
        ONLY ADDOING ONE DATA POINT, YOU NEED TO DO AS IN THE EXAMPLE.
        '''
        if kwargs.get('show_deprecation_warning',False):
            import warnings
            warnings.simplefilter('always')
            warnings.warn('This function will soon be moved to be private;'+
                ' it is recommended to use self.create() for dataset creation instead.',PendingDeprecationWarning)
            warnings.simplefilter('default')
        #

        import numpy as np
        from calcom.io import CCDataPoint
        from calcom.io import CCList

        import copy
        from calcom.utils import type_functions as tf

        verb = kwargs.get('verbosity',0)

#        import pdb
#        pdb.set_trace()

        # Before going in earnest, check if the values are numeric
        # and update self.attrs. No error checking is done here.
        firstpt = attrvals[0]
        if tf.is_list_like(firstpt):
            for j,av in enumerate(firstpt):
                setattr(self.attrs[newattrs[j]],'type',type(av))
                setattr(self.attrs[newattrs[j]],'is_numeric',tf.is_numeric(av))
            #
        else:
            # We're assuming that there's only a single data point, and
            # they didn't wrap it in an extra list beforehand; that is,
            # they input [1,2,3] rather than [[1,2,3]].
            for j,av in enumerate(newattrs):
                setattr(self.attrs[newattrs[j]],'type',type(av))
                setattr(self.attrs[newattrs[j]],'is_numeric',tf.is_numeric(av))
            #
        #

        ######################################################

        # Lettuce pray
        firstpt = attrvals[0]
        if verb>0: print('Reading in datapoints...')
        if type(firstpt) in [tuple,list,np.ndarray]:
            for i,row in enumerate(dataarray):
                ccdatum = CCDataPoint(row)

                newattrs_copy = copy.deepcopy([self.attrs[natt] for natt in newattrs])

                ccdatum.set_attrs(newattrs_copy,attrvals[i])

                # Generate an ID.
                idd = self._generate_id(**kwargs)
                setattr(ccdatum,'_id',idd)
                self.ids.append( idd )
                self.data.append( ccdatum )

                # Set '_loaded' to true; this is always true for the session
                # in which the data is being created.
                setattr(ccdatum,'_loaded',True)

                if verb>1:
                    if len(dataarray)>100:
                        if 100*i%len(dataarray)==0:
                            print('\tLoaded datapoint %i of %i.'%(i+1,len(dataarray)))
                    else:
                        print('\tLoaded datapoint %i of %i.'%(i+1,len(dataarray)))
            #
        elif len(np.shape(attrvals)) == 1:
            # Apply the single attrval vector to all points.
            for i,row in enumerate(dataarray):
                ccdatum = CCDataPoint(row)
                # ccdatum.set_attrs(self.attrs, [self.attrs[natt] for natt in newattrs],attrvals)
                newattrs_copy = copy.deepcopy([self.attrs[natt] for natt in newattrs])

                ccdatum.set_attrs(newattrs_copy,attrvals)

                # Generate an ID.
                idd = self._generate_id(**kwargs)
                setattr(ccdatum,'_id',idd)
                self.ids.append( idd )
                self.data.append( ccdatum )

                if verb>1:
                    if len(dataarray)>100:
                        if 100*i%len(dataarray)==0:
                            print('\tLoaded datapoint %i of %i.'%(i+1,len(dataarray)))
                    else:
                        print('\tLoaded datapoint %i of %i.'%(i+1,len(dataarray)))
            #
        #
        if verb>0: print('done.')

        return
    #

    def add_variable_names(self,varnames,**kwargs):
        '''
        Dumps the input list of strings into self.variable_names.

        Input:
            varnames: A list of strings
        Optional input:
            vardescrs: A list of strings describing each variable. If not
                specified, this list is not populated.
        Output:
            None.
        '''

        from calcom.io import CCList

        n = len(varnames)
        vardescrs = kwargs.get('vardescrs',False)

        self.variable_names = CCList(varnames)
        if vardescrs!=False:
            self.variable_descrs = CCList(vardescrs)
        #
        return
    #

    def create(self, datapoints, metadata=None, **kwargs):
        '''
        Populates the CCDataSet in a single line.
        Datapoints row vectors in R^d, assumed to correspond
        to each datapoint.

        If supplied, it's assumed the metadata is well-structured:
            Row 0 is the name of each attribute;
            Rows 1 through n contain the corresponding attribute values
                for datapoints 0 through n-1.

        Inputs:
            datapoints: list-of-lists or array of shape n-by-d, of n datapoints
                in d dimensions
        Optional inputs:
            metadata: list-of-lists or array of shape either n-by-m, or
                (n+1)-by-m, of m attributes for each datapoint.

            variable_names: a list length d containing names for the variables
                in the dataset.
                (default: ['var%i'%str(i).zfill(int(np.log10(d))) for i in range(d)])
            collect_study_attrs: Boolean; whether to post-process the dataset
                and remove attributes which have the same value for all
                datapoints. (default: False)
            missing_header: Boolean; indicating if input metadata is missing
                its header providing attribute names. If True, then dummy
                names 'attr0', 'attr1', ... are used for the attributes, and
                it's assumed metadata starts from row 0 rather than row 1. (default: False)
            attrdescrs: list of strings. Long-form descriptions for each attribute.
                (default: list of empty strings)

        Outputs: None. The CCDataSet is populated in-place.
        '''
        import numpy as np

        n,d = np.shape(datapoints)
        ndigits_vars = np.nanmax( [1,int(np.floor(np.log10(d)))] )

        if type(metadata)==type(None):
            # No metadata?
            m = 0
            attrnames = []
            attrvals = [[] for _ in range(n)]
            attrdescrs = []
        else:
            ncheck,m = np.shape(metadata)
            attrdescrs = kwargs.get('attrdescrs', ['' for _ in range(m)])
            if ncheck==n+1:
                attrnames = metadata[0]
                attrvals = metadata[1:]
            elif ncheck==n and kwargs.get('missing_header',False):
                ndigits_attrs = np.nanmax( [1,int(np.floor(np.log10(m)))] )
                attrnames = ['attr%s'%str(i).zfill(ndigits_attrs) for i in range(m)]
                attrvals = metadata
            else:
                raise ValueError('The length of the metadata (%i) and the length of the datapoints (%i) do not agree.'%(ncheck,n)+
                    ' Please check the inputs and ensure you have set any optional flags correctly.')
        #
        
        # Check attrnames for redundancies. For now, raise an exception.
        # In the future, silently (or non-silently) rename attributes 
        # to make unique. Non-uniqueness not allowed for multiple reasons.
        if len(np.unique(attrnames))<len(attrnames):
            raise ValueError('CCDataSet() requires attribute names to be unique!')
        #
        
        self.add_attrs(attrnames,attrdescrs=attrdescrs, show_deprecation_warning=False)
        self.add_datapoints(datapoints, attrnames, attrvals, show_deprecation_warning=False)

        self.add_variable_names( kwargs.get('variable_names', ['var%s'%str(i).zfill(ndigits_vars) for i in range(d)]) )

        if kwargs.get('collect_study_attrs', False):
            self.cleanup_attrs(**kwargs)
        #

        return
    #

    #######################################################################

    def generate_labels(self,attrname,**kwargs):
        '''
        Loops over the data, generating a minimal set of labels on that
        attribute, then returns a set of integer labels, and the dictionary
        mapping back to the original attribute values.

        Inputs:
            attrname: String indicating the attribute to generate labels on.
        Outputs:
            labels: array containing the attribute name's labels for each point.

        Optional inputs:

            idx: which subset of the data to look at. Note that the
                original ordering is lost once the data and labels are sliced.
                Defaults to np.arange(0, len(self.data)).
            keep_nan: whether to keep `None` and numpy.nan attribute values or not.
                If set to True, the function will behave naively and return
                    all labels in the associated index list.
                If set to False, bad values will be filtered out and
                    the function will return the subset of the labels
                    with "good" values and the corresponding filter (boolean array)
                    indicating the locations of "good" datapoints.
                    (default: True)
        '''
        if kwargs.get('debug', False):
            import pdb
            pdb.set_trace()
        #
        import numpy as np
        from calcom.io import CCList


        n = len(self.data)
        if ( 'idx_list' in list(kwargs.keys()) ) and ( 'idx' not in list(kwargs.keys()) ):
            # re-instate backwards support
            idx = kwargs['idx_list']
        else:
            idx = kwargs.get('idx', np.arange(0,n, dtype=np.int64))
        #

        attrvalues = self.get_attrs(attrname, idx=idx)

        nanlocs = (np.array(attrvalues)!=np.array(attrvalues))
#        wherenan = np.where(np.isnan(attrvalues))[0]
#        nanlocs = np.zeros(len(idx), dtype=bool)
#        nanlocs[wherenan] = True

        nonelocs = np.array([type(ai)==type(None) for ai in attrvalues])
        badlocs = np.where(np.logical_or(nanlocs,nonelocs))

        filter = np.array( np.ones(len(idx)), dtype=bool )
        if not kwargs.get('keep_nan', True):
            # filter = np.setdiff1d(idx, idx[badlocs])
            # Needs to be done this way to preserve ordering associated with idx.
            filter[badlocs] = False
        #

        labels = np.array(attrvalues)[filter]

        if kwargs.get('keep_nan', True):
            return labels
        else:
            return labels, filter
        #
    #

    def generate_data_matrix(self,**kwargs):
        '''
        Outputs an numpy.ndarray of the data matrix, to be used in
        classification algorithms.

        Inputs: none
        Outputs: numpy.ndarray of two dimensions.

        Optional inputs (kwargs):
            idx: List of integers indicating the subset of the data
                to pull.
            features: String/List of integers, indicating the specific
                dimensions in the data to pull.
            attr_list: List of strings indicating attribute values to append to
                the raw data. The attributes are concatenated in the order they
                appear in the argument list, whether they are scalars or
                vectors.
                Default: empty list
            use_data: Boolean.If False, the CCDataPoint is ignored, and only
                the values referenced in attr_list are used.
                Default: True

        If a list of integers is specified, only the specified subset of the
        data is pulled.

        Examples:
            # Full dataset pulled
            data0 = ccd.generate_data_matrix()
            # Only data points 2,4,6 pulled
            data1 = ccd.generate_data_matrix(idx=[2,4,6])
        '''
        import numpy as np
        from calcom.io import CCList
        from calcom.utils import type_functions as tf

        from calcom.io.loadsave_hdf import load_CCDataPoint

        n = len(self.data)

        if ( 'idx_list' in list(kwargs.keys()) ) and ( 'idx' not in list(kwargs.keys()) ):
            # re-instate backwards support
            idx = kwargs['idx_list']
        else:
            idx = kwargs.get('idx', np.arange(0,len(self.data), dtype=np.int64))
        #

        if ( 'feature_set' in list(kwargs.keys()) ) and ( 'features' not in list(kwargs.keys()) ):
            # re-instate backwards support
            features = kwargs['feature_set']
        else:
            features = kwargs.get('features', [])
        #

#        features = kwargs.get('features', [])

        attr_list = kwargs.get('attr_list', CCList())
        use_data = kwargs.get('use_data', True)

        #

        if tf.is_string_like(features):
            if len(features)>0:
                if features not in self.feature_sets:
                    print('Warning: feature set %s not found in CCDataSet. Using all features.'%features)
                    features=[]
                else:
                    features = self.feature_sets[features]
                #
            else:
                features=[]
            #
        #

        output = CCList()

        for i in idx:
            elem = CCList()
            d = self.data[i]
            if not d._loaded:
                d = load_CCDataPoint(self.fname, d._id)
                d._loaded = True
                self.data[i] = d
            #

            if use_data:
                if len(features)==0:

                    # elem.append( self.data[i].flatten() )   # Row-wise concatenation if it is a 2D array.
                    elem.append( d )
                else:
                    # Need to break into cases to do the slicing properly.
                    # Only two-dimensional arrays supported. TODO: is there a
                    # way to slice on the first dimension (python counting)
                    # if the order of the array is unknown? Probably won't be needed.
                    if len(np.shape( d ))==2:
                        elem.append( d[:,features] )
                    elif len(np.shape( d ))==1:
                        elem.append( d[features] )
                    #
                #
            #
            for attr in attr_list:
                elem.append( getattr(getattr(d,attr),'value') )
            #

            # elem = np.hstack(elem)
            output.append( np.hstack(elem) )
        #

        return np.array(output)
    #

    def generate_classification_problem(self,attrname,**kwargs):
        '''
        Calls self.generate_data_matrix() and self.generate_labels() and
        returns a data matrix and labels. Keyword arguments (kwargs)
        to those functions are passed forward.

        Inputs:
            attrname: A string indicating the attribute on which the labels are
                created.
        Outputs:
            Depends on the optional input make_dict. If make_dict==True, then
            there are three outputs:
                data - a numpy array; the data matrix;
                labels - a numpy array of the integer labels

        Optional inputs:
            All optional inputs are passed to self.generate_data_matrix() and
            self.generate_labels(); see those functions for options.  Of
            particular importance is the optional input:

            keep_nan: whether to keep `None` attribute values or not.
                If set to `True`, None values will be replaced by `-1`
                If set to `False`, None values will be ignored for both labels
                and data
                (Default: `True`)

        '''

        import numpy as np

        n = len(self.data)

        idx = kwargs.get('idx', np.arange(0,n, dtype=np.int64))

        keep_nan = kwargs.get('keep_nan', True)

        if keep_nan:
            labels = self.generate_labels(attrname,**kwargs)
        else:
            labels, keep_idx = self.generate_labels(attrname,**kwargs)
        #

        # TODO: there's probably weird conflicts when keep_nan==False
        # and the user uses this function. Data and labels arrays might
        # not end up being the same. Not completely sure how to fix.
        # edge case, though; may want to consider
        # dropping that functionality altogether.

        # Make sure that len(labels) and len(data) is same

        data = self.generate_data_matrix(**kwargs)

        return data,labels
    #

    def _find_one(self,attrname,attrvalue,**kwargs):
        '''
        Given an attribute and a specified value, returns a list of integer
        pointers to the data which have that value.

        Currently a hidden function called by the public-facing .find().
        Kept for now to maintain functionality of .find() just iteratively
        calling ._find_one().

        Inputs:
            attrname: A string indicating the desired attribute.
            attrvalue: Either a single attribute value to search for, or a list
                of values to search for.

        Optional inputs:
            idx: A list of integers. If provided, the search is only done
                on a subset of the data. Used to more efficiently search over
                multiple attributes, implemented in
                self.find().
            regex: Boolean. If True, then the attrvalue is interpreted as a
                regular expression to be matched. Default: False

        Outputs:
            idx: An list of integers indicating the location of the data.

        Examples:
            # Pulling a subset of the data whose "StudyID" value is "Duke".
            idx = ccd._find_one('StudyID','Duke')
            data = ccd.generate_data_matrix(idx=idx)
        '''
        import numpy as np
        from calcom.io import CCList

        import re
        from calcom.utils import type_functions as tf

        n = len(self.data)

        if ('idx_list' in kwargs) and ('idx' not in kwargs):
            # just keep it for backward compatibility; don't advertise it.
            rows = kwargs.get('idx_list', np.arange(0,len(self.data), dtype=np.int64))
        else:
            rows = kwargs.get('idx', np.arange(0,len(self.data), dtype=np.int64))
        #

        regex = kwargs.get('regex', False)

        # Check if they're asking for something we have.
        anames_case_insensitive = [a.lower() for a in self.attrnames]
        if (attrname.lower() not in anames_case_insensitive) and (not regex):
            raise ValueError("Requested attribute is not in the CCDataSet.")
        #

        # Correct the attribute name if the cases are screwed up.
        if attrname not in self.attrnames:
            ii = np.where([attrname.lower()==a.lower() for a in self.attrnames])[0]
            attrname = self.attrnames[ii][0]    # recall: CCList.__getitem__ always returns a CCList.
        #

        # This won't work for vector-valued attributes right now.
        # Don't know if this will ever be a problem with the current paradigm.
        # Vector-valued attrs are possible, but only used to append to
        # a data matrix at the moment.
        if not tf.is_list_like(attrvalue):
            attrvalue = CCList([attrvalue])
        #

        idx_out = CCList()
        for av in attrvalue:
            if tf.is_string_like(av):
                av = av.lower()
                av = av.strip() # remove leading/trailing white space
                if not regex:
                    # Use regex anyway, but look only for exact matches.
                    # Praying that users don't have any of ()[]*\^$
                    # in their attribute values.
                    #
                    # This really isn't safe - should just do an
                    # exact string comparison.
                    av = '^'+av+'\Z'

                    # Need to insert escape characters in the situation
                    # that the attribute has special characters...

                    specials = ['(', ')', '[', ']'] # there are probably more.
                    for s in specials:
                        av = av.replace(s, "\\"+s)
                    #
                #
                reprog = re.compile(av)
                for i in rows:
                    subj = self.data[i]
                    val = getattr(subj,attrname).value
                    if type(val)==type(None):
                        continue
                    #
                    val = val.lower()
                    if reprog.match(val):
                        idx_out.append(i)
                    #
                #
            elif type(av)==type(None):
                # Specialty case; we can't handle this as we would normally,
                # since None is not equivalent to anything else (even itself).
                for i in rows:
                    subj = self.data[i]
                    val = getattr(subj,attrname).value
                    if type(val)==type(av):
                        idx_out.append(i)
                    #
                #
            else:
                for i in rows:
                    subj = self.data[i]
                    val = getattr(subj,attrname).value
                    if type(val)==type(None):
                        continue
                    #
                    if val == av:
                        idx_out.append(i)
                    #
                #
            #
        #

        # Unique-ize the output. Can't hurt.
        idx_out = np.unique(idx_out)

        return idx_out
    #

    def find(self,query,*args,**kwargs):
        '''
        Finds a list of datapoints which satisfies the query. A query
        essentially takes the form of a collection of "and" statements,
        defined in terms of a dictionary whose keys are attributes,
        and whose values are some criteria which must be met.
        For example, the following:

            query = {'attr1': [2,4,5], 'attr2': 'h1n1|h3n2'}

        searches for all data which both takes on values 2, 4, or 5 for 'attr1',
        and matches 'h1n1' or 'h3n2'. Basic regular expressions are supported
        for string attributes; otherwise lists denote "or" statements.

        A single query is supported without using a dictionary, as well; e.g.,

            self.find('attr1', [2,4,5],**kwargs)

        achieves the same thing as

            self.find({'attr1':[2,4,5]}).

        Inputs:
            query : a dictionary, as described above.
        Outputs:
            idx : list of pointers for which the collection of queries is true.

        Optional inputs:
            idx : list of pointers for the subset of data on which to look.
                Note that pointers are always relative to the full dataset, so
                that the output "idx" will be a subset of this input list.
                Default: np.arange(0,len(self.data)).
            regex: Boolean. If True, then the string-type values are interpreted as a
                regular expression to be matched. Defaults to True.

        '''
        from calcom.utils import type_functions as tf
        if kwargs.get('debug',False):
            import pdb
            pdb.set_trace()
        #

        # if not (isinstance(query,dict) or tf.is_list_like(query)):
        if not (isinstance(query,dict) or tf.is_list_like(query)):
            # assumed "scalar" query; cast to dictionary
            # anticipating args[0] is the corresponding value.
            query = {query: args[0]}
        #

        # For now, just cast dictionary to list and use the old code.
        if isinstance(query,dict):
            attrpairs = list(query.items())
        else:
            attrpairs = query
        #

        for i,pair in enumerate(attrpairs):
            attrname = pair[0]
            attrvalue = pair[1]
            if i==0:
                idx = self._find_one(attrname,attrvalue,**kwargs)
            else:
                kwargs['idx'] = idx
                idx = self._find_one(attrname,attrvalue,**kwargs)
            #
        #
        return idx
    #

    def generate_attr_from_queries(self,attrname,attrpairs_dict,**kwargs):
        '''
        Given a collection of queries that you would pass to
        self.find, create a new attribute on the *entire
        dataset* based on the positive matches for each of the queries.

        This function DOES NOT check if a data point satisfies multiple
        queries. The current convention is that datapoint is assigned the key
        for the first attrpairs_dict value satisfied (the searches are done in
        ascending order of the sorted keys).

        If a list idx is specified, all data not in the list "idx" are
        automatically assigned None for the new attribute.

        Inputs:
            attrname: String. The name for the new attribute.

            attrpairs_dict: A dictionary of attrpairs. The keys will be the
                attribute values assigned to each of the seraches.  The values
                corresponding to the keys are expected to be compatible with
                self.find().

        Outputs:
            None. The new attribute is appended to the data, and onto
                self.attrs and self.attrnames.
        Optional inputs:
            attrdescr: String. Long description for the new attribute.
            idx: list-like. Pointers to data to restrict the searches to.
            verbosity: integer. If positive, text is output describing details.
                (default: 0)

            All other optional arguments are passed along to calls to
            self.find().

        '''
        import numpy as np

        attrdescr = kwargs.get("attrdescr", "")

        if ('idx_list' in kwargs) and ('idx' not in kwargs):
            # just keep it for backward compatibility; don't advertise it.
            idx = kwargs.get('idx_list', np.arange(0,len(self.data), dtype=np.int64))
        else:
            idx = kwargs.get('idx', np.arange(0,len(self.data), dtype=np.int64))
        #
        verbosity = kwargs.get('verbosity', 0)

        attr_out = np.array([None for i in range(len(self.data))], dtype='O')

        active_set = list(idx)

        keys = list(attrpairs_dict.keys())
        keys.sort()

        kwargs_new = dict(kwargs)

        for k in keys:
            ap = attrpairs_dict[k]
            idxs = self.find(ap,**kwargs_new)
            if len(idxs)==0:
                if verbosity>0: print('Warning: no data was found matching criteria for key %s. Skipping.'%k)
                continue
            #
            attr_out[idxs] = k
            active_set = np.setdiff1d(active_set, idxs)
            kwargs_new['idx'] = active_set
        #

        self.append_attr(attrname,attr_out,attrdescr=attrdescr, is_derived=True)

        return
    #

    def get_attrs(self,attrname,**kwargs):
        '''
        Returns a list of all values associated with attrname. Non-unique.

        Inputs:
            attrname: String indicating the attribute to look along.
        Outputs:
            attrvalues: List of attribute values associated with attrname.

        Optional inputs:
            idx: List/np.ndarray of integers indicating the subset of the
                data to restrict to. Defaults to np.arange(0,len(self.data)).

        '''
        import numpy as np
        from calcom.io import CCList,CCDataAttr

        n = len(self.data)
        if ('idx_list' in kwargs) and ('idx' not in kwargs):
            # just keep it for backward compatibility; don't advertise it.
            idx = kwargs.get('idx_list', np.arange(0,len(self.data), dtype=np.int64))
        else:
            idx = kwargs.get('idx', np.arange(0,len(self.data), dtype=np.int64))
        #

        attrvals = CCList()
        # for point in self.data:
        for i in idx:
            point = self.data[i]
            attr = getattr(point,attrname)
            if isinstance(attr, CCDataAttr):
                attr = getattr(attr,'value')
            attrvals.append(attr)
        #
        return np.array(attrvals)
    #

    def partition(self,attrname,**kwargs):
        '''
        Given an attribute name, returns information to partition the entire
        set by the attribute values.

        Inputs:
            attrname: String indicating attribute to partition along.
        Optional inputs:
            idx: list of integers indicating subset of data to generate.
                Indexing is preserved relative to the entire dataset.
        Outputs:
            equivclasses: Dictionary, with keys being attrvals, and values
                being lists of indexes of the dataset taking that value.

        '''

        import numpy as np
        from calcom.io import CCList

        if ('idx_list' in kwargs) and ('idx' not in kwargs):
            # just keep it for backward compatibility; don't advertise it.
            idx = kwargs.get('idx_list', np.arange(0,len(self.data), dtype=np.int64))
        else:
            idx = kwargs.get('idx', np.arange(0,len(self.data), dtype=np.int64))
        #

        # Correct the attribute name if the cases are screwed up.
        if attrname not in self.attrnames:
            ii = np.where([attrname.lower()==a.lower() for a in self.attrnames])[0]
            attrname = self.attrnames[ii][0]    # recall: CCList.__getitem__ always returns a CCList.
        #

        attrvals = CCList()
        equivclasses = {}
        # for i,datum in enumerate(self.data):
        for i in idx:
            datum = self.data[i]

            atv = getattr(datum,attrname).value
            if atv not in attrvals:
                attrvals.append(atv)
                equivclasses[atv] = CCList([i])
            else:
                equivclasses[atv].append(i)
            #
        #
        # return attrvals,equivclasses
        return equivclasses
    #

    def summarize_attr(self,attrname,**kwargs):
        '''
        Calls self.partition and self.attrs[attrname] and prints
        a summary of the information gathered.

        Inputs:
            attrname: string. Which attribute to look at.
        Outputs:
            None. However, the summary is printed to screen.

        Optional inputs:
            w : integer. A string formatting parameter; a cutoff for very long
                strings.
        '''
        from calcom.io import CCList
        import numpy as np

        w = kwargs.get('w',20)  # Formatting parameter

        # Correct the attribute name if the cases are screwed up.
        if attrname not in self.attrnames:
            ii = np.where([attrname.lower()==a.lower() for a in self.attrnames])[0]
            attrname = self.attrnames[ii][0]    # recall: CCList.__getitem__ always returns a CCList.
        #

        eq = self.partition(attrname,**kwargs)
        uvals = list(eq.keys())

        attr = self.attrs[attrname]

        if None in uvals:
            uvals.remove(None)
            uvals.sort()
            uvals.append(None)
        else:
            uvals.sort()
        #

        maxl = max([len(str(u)) for u in uvals])

        print('\nAttribute: %s.'%attrname[:w])
        if attr.long_description:
            print('\n%s'%attr.long_description)

        print('\nGeneral attribute properties:')
        for t in ['is_numeric','is_derived']:
            print('\t%s : %s'%(t,getattr(attr,t)))
        #

        print('\nThe partitioning of the data on this attribute:')
        for u in uvals:
            val = CCList(eq[u])
            u = (str(u) + ' '*maxl)[:maxl]
            print('\t%s : %s'%(u, str(val)))
        #
        print('')
        return
    #

    def save(self,filename,**kwargs):
        '''
        Saves the current dataset to the given filename by calling
        calcom.io.save_CCDataSet(self,filename,**kwargs).

        Inputs:
            filename: String, indicating the location and name filename
        '''

        # New: do a minimal amount of sanity checking.
        # Identifiers shouldn't have slashes in them; HDF may
        # freak out if they do.
        idvs = self.get_attrs('_id')
        if any( ['/' in iv for iv in idvs] ):
            raise ValueError('Datapoint identifiers cannot contain forward slashes "/".'+
                'Either use integer identifiers or review the data used to generate the identifiers.')
        #

        from calcom.io import save_CCDataSet
        save_CCDataSet(self,filename,**kwargs)
        return
    #

    def load(self,filename,**kwargs):
        '''
        Loads a dataset from the given filename by calling
        calcom.io.load_CCDataSet(self,filename,**kwargs).

        Assumes the current data structure is empty. Will probably overwrite
        any existing data here.

        Inputs:
            filename: String, indicating the location and name filename
        Optional inputs:
            preload (default: True):
                If boolean:
                    If True, the entire dataset is populated immediately.
                    If False, data is loaded only when the user explicitly asks for
                    it; datapoints are populated with only metadata immediately,
                    and the actual data is loaded on the fly with a supplementary
                    function.

                _Planned for future_:
                If list-like:
                    The collection of points in [self.data[i] for i in preload]
                    only is loaded, assuming everything else in the dataset is the same.
        '''

        from calcom.io import load_CCDataSet

        self = load_CCDataSet(self,filename,**kwargs)

        try:
            # We'd prefer absolute paths to avoid the 
            # jittery user jumping around folders then trying to save 
            # aspects of the CCDataSet.
            import os
            self.fname = os.path.abspath( filename )
        except:
            # but fall back on simpler behavior if for some reason "os" misbehaves.
            self.fname = filename
        #
        
        # for convenience... some may think to look here.
        self.__file__ = self.fname
        
        if kwargs.get('print_about',True):
            self.about()
        #
        
        return
    #

    def generate_metadata(self,**kwargs):
        '''
        Generates a table of the metadata associated with all data.
        Missing values are replaced with "None".

        Inputs:
            None
        Optional inputs:
            format: string; one of 'df', 'str', 'tuple'. Modifies the output. (default: 'df')

            verbosity: level of output. (default: 0)
            idx: pointers of data to generate metadata for (default: all data)
            sortby: string, or list of strings, indicating attributes to pre-sort
                the data. (default: '_id'). Priority reads from left to right
                (sorted first by sortby[0], then sortby[1], and so on).
                Powered by a call to self.sort_by_attrvalues().
            save_to_disk: Boolean. Directly saves the generated table to h5 file.
                Overwrites anything else there. (default: False)

            In the case format=='str', arguments for the type of output are considered:
                delimiter: string; what type of delimiter to be used in the case that
                    format=='str'. (default: '\t'; i.e., tab-separated)
                newline: string; how to create a newline. (default: '\n').

        Outputs:
            A table of the metadata. Its format depends on optional inputs:
                'df': A pandas DataFrame object, whose rows correspond to
                    CCDataPoint()s and columns correspond to attributes/metadata.
                'string': A raw string which can be exported to file via:
                    output = self.generate_metadata(...)
                    f = open(fname,'w')
                    f.write(output)
                    f.close()
                'tuple': A tuple of the raw output needed to reconstruct the table;
                        output[0]: row labels (sorted idx)
                        output[1]: column labels (self.attrnames, leading with id)
                        output[2]: interior of table, as a list of lists.

                    Primary use is to populate self._metadata_table. This
                    functionality is for user to load metadata and access
                    subsets of data without loading the entire CCDataSet.
                    (We don't want pandas as a hard dependency and we
                    need a raw output for HDF anyways.)

        '''
        import numpy as np

        format = kwargs.get('format', 'df')
        verb = kwargs.get('verbosity', 0)

        sortby = kwargs.get('sortby', '_id')
        save_to_disk = kwargs.get('save_to_disk', False)

        if ('idx_list' in kwargs) and ('idx' not in kwargs):
            # just keep it for backward compatibility; don't advertise it.
            idx = kwargs.get('idx_list', np.arange(0,len(self.data), dtype=np.int64))
        else:
            idx = kwargs.get('idx', np.arange(0,len(self.data), dtype=np.int64))
        #

        if type(sortby)==str:
            sortby = [sortby]
        #
        order = self.lexsort(sortby)

        idx = np.array(idx)[order]

        # What are our delimiter and newline characters for string output?
        dl = kwargs.get('delimiter', '\t')
        nl = kwargs.get('newline', '\n')

        columns = self.attrnames
        try:
            columns.remove('_id')
        except:
            pass
        #

        # This should be gone by now... but just in case...
        try:
            columns.remove('id')
        except:
            pass
        #


        # Generate the raw table
        table = []
        for i in idx:
            d = self.data[i]
            row = [d._id] + [getattr(d,attr).value for attr in columns]
            table.append(row)
        #

        if format=='df':
            # EZ
            import pandas
            output = pandas.DataFrame(data=table, index=idx, columns=['_id'] + columns)
        elif format=='str':
            # Only reason to do this ourselves is if the user doesn't want to
            # deal with pandas at all; since pandas DataFrames have their own
            # export to csv option.
            output = dl + '_id' + dl
            output += dl.join(columns) + nl

            for i,row in enumerate(table):
                strrow = str(idx[i])
                for j,e in enumerate(row):
                    try:
                        strrow += dl + str(e)
                    except:
                        raise TypeError('ERROR: Unable to cast attribute %s for datapoint %s to string.'%(columns[j], d._id))
                    #
                #
                strrow += nl
                output += strrow
            #
        elif format=='tuple':
            # User can call this function with extra arguments if they prefer
            # a particular ordering of the data. Otherwise save_CCDataSet calls
            # the function with this flag just to populate the thing.
            output = (idx, ['_id'] + list(columns), table)
            self._metadata_table = output
        #

        if save_to_disk:
            # TODO - MOVE THIS OVER TO loadsave_hdf.py AS ITS OWN FUNCTION
            import h5py
            h5f = h5py.File(self.fname)
            h5f_ccd = h5f['CCDataSet']
            if 'metadata_table' not in h5f_ccd.keys():
                h5_metadata_table = h5f_ccd.create_group('metadata_table')
                h5_metadata_table.create_dataset('rows', data=idx, compression="gzip")
                h5_metadata_table.create_dataset('columns', data=np.array(['_id']+columns,dtype=np.string_), compression="gzip")
                h5_metadata_table.create_dataset('metadata', data=np.array(table, dtype=np.string_), compression="gzip")
            else:
                # BUG: CANNOT OVERWRITE OLD TABLE. DO NOT KNOW THE FUNCTIONALITY;
                # CAN'T BE BOTHERED TO LOOK IT UP RIGHT NOW. TOO LATE AT NIGHT.
                h5_metadata_table = h5f_ccd['metadata_table']
                h5_metadata_table['rows']=idx
                h5_metadata_table['columns']=np.array(['_id']+columns,dtype=np.string_)
                h5_metadata_table['metadata']=np.array(table, dtype=np.string_)
            #

            h5f.close()
        #

        return output
    #

    def generate_ccd_by_attr_values(self,attrpairs,**kwargs):
        '''
        Generates a new CCDataSet where the entries satisfy the conditions in
        attrpairs. Works by calling find(attrpairs) and then
        popping the complement from a copy.deepcopy of the dataset.

        One usage is for when you want to do a ccexperiment restricting to a
        subset of attributes separate from the classification and partition
        attributes.

        Inputs:
            attrpairs: A list of lists compatible with find()
        Outputs:
            ccdnew: A new CCDataSet which only has the data from satisfying
                conditions in the attrpairs.
        Optional inputs:
            All keyword arguments are passed to find().
        '''
        import warnings
        warnings.simplefilter('always')
        warnings.warn('This function may be deleted in a future version.', FutureWarning)
        warnings.simplefilter('default')

        import copy
        from calcom.io import CCList

        idxs = self.find(attrpairs,**kwargs)

        ccdnew = CCDataSet()
        attrnames = CCList(self.attrs.keys())
        attrnames.remove('_id')

        attrdescrs = CCList([self.attrs[a].long_description for a in attrnames])
        ccdnew.add_attrs(attrnames,attrdescrs)

        data = CCList()
        attrvalues = CCList()
        for i in idxs:
            d = self.data[i]
            data.append(np.array(d))
            attrvalues.append( CCList([getattr(d,name).value for name in attrnames]) )
        #

        ccdnew.add_datapoints(data,attrnames,attrvalues)

        ccdnew.variable_names = CCList(self.variable_names)
        ccdnew.variable_descrs = CCList(self.variable_descrs)
        ccdnew.feature_sets = copy.deepcopy(self.feature_sets)
        ccdnew.study_attrs = copy.deepcopy(self.study_attrs)

        return ccdnew
    #

    def lexsort(self,attrnames,**kwargs):
        '''
        Perform an indirect sort of data_points using a sequence of attrnames
        by utilizing numpy.lexsort(). Sorts by a list of attributes, ordered
        from left to right, and returns the indexing associated with that
        ordering.

        Inputs:
            attrnames: list of attribute names to sort the dataset by Example-
                if attrnames is ['a','b','c'] then first sort by 'a'; for same
                values of 'a' sort by 'b'; for same values of 'b' sort by 'c'

        Outputs:
            Sorted indices according to the attribute names

        Optional inputs:
            idx: list of pointers of the subset of data upon which to
                do the sorting. NOTE: if this is specified, the output
                sorting is in terms of the same pointers, NOT relative
                to the entries of idx. That is, if the lexsort
                of idx=[2,5,6,8] results in an ordering idx=[6,5,2,8],
                the output of this function is [6,5,2,8], NOT [2,1,0,3].
            relative_ordering : If True, the relative ordering discussed
                previously is returned instead. (Default: False)
        '''

        import numpy as np
        from calcom.io import CCList

        idx = kwargs.get('idx',np.arange(0,len(self.data),dtype=np.int64))

        l = CCList([CCList() for i in attrnames])
        for i, attrname in enumerate(attrnames[::-1]): #Notice we are taking the reverse order to maintain sorting priority
            l[i] = self.get_attrs(attrname, idx=idx)

        rel_ordering = np.lexsort(tuple(l))
        if kwargs.get('relative_ordering',False):
            return rel_ordering
        else:
            return np.array(idx)[rel_ordering]
    #
#

