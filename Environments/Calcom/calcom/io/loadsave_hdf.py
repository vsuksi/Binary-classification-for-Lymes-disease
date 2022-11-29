def save_feature_set(filename, feature_set_name, feature_set, **kwargs):
    '''
    Saves a feature_set in the given hdf5 file.

    Inputs:
        filename: String, indicating the path to hdf5 file, the file must exist
        feature_set_name: String, name of the feature set
        feature_set: List of integers.

    Optional inputs:
        verbosity: Integer, indicating level of output. Default: 0
    '''
    import h5py
    from calcom.utils import type_functions as tf

    if type(feature_set_name)!=str:
        raise ValueError("Expected a string")

    if not tf.is_list_like(feature_set):
        raise ValueError("Expected a list of integers")

    verb = kwargs.get('verbosity',0)
    if verb>0: print('Begin saving feature set `%s` to disk under `%s`'%(feature_set_name,filename))
    # make sure to use 'r+' mode to write to existing file. 'w' mode will create a new empty file.
    with h5py.File(filename, 'r+') as f:
        h5_FeatureSets = f['CCDataSet']['feature_sets']
        # if a feature set with similar name already exists, delete it.
        # if h5_FeatureSets[feature_set_name]: # This switch throws a KeyError and doesn't continue for me.
        if feature_set_name in h5_FeatureSets:
            if verb>0: print("Overwriting existing feature set `%s`"%feature_set_name)
            del h5_FeatureSets[feature_set_name]
        # save the new feature set
        h5_FeatureSets.create_dataset(feature_set_name, data=feature_set)
    #
#

def save_CCDataSet(oCCDataSet, filename, **kwargs):
    '''
    Saves a CCDataSet object to the given filename in a hdf format.
    Hierarchy of HDF:
        CCDataSet
        |--CCDataPoint1 (names using the IDs)
        |--CCDataPoint2
        |--
        |--CCDataPointN
    Each CCDataPoint saved as seperate datasets and each have respective attributes

    Additionally, '/CCDataset' Group has the datasets:
        'study_info'
        'feature_sets'
        'metadata_table' (generated here if it doesn't exist in oCCDataSet._metadata_table

    Inputs:
        filename: String, indicating the location and name filename

    Optional inputs:
        verbosity: Integer, indicating level of output. Default: 0
        compression_level: Integer[0-9], indicating level of compression. Default: 0, meaning no compression
        debug: developer debug flag. if there's a pdb.set_trace() lying
            around, it will get triggered if this is True. (default: False)
    '''

    import h5py
    import numpy as np
    import json
    from calcom.utils import type_functions as tf

    verb = kwargs.get('verbosity',0)
    compression_level = kwargs.get('compression_level',0)
    debug = kwargs.get('debug', False)

    if not oCCDataSet._metadata_table: # does this exist?
        meta_rows, meta_columns, meta_data = oCCDataSet.generate_metadata(format='tuple')
    else:
        meta_rows, meta_columns, meta_data = oCCDataSet._metadata_table
    #

    if verb>0: print('Begin saving %s to disk under %s'%(type(oCCDataSet),filename))
    with h5py.File(filename, 'w') as f:
        f.attrs['version'] = oCCDataSet.version
        h5_CCDataSet = f.create_group('CCDataSet')

        if verb>0: print('Saving the about/readme.')
        f.attrs['about'] =  oCCDataSet._about_str
        if verb>1: print('saved.')


        if verb>0: print('Saving top-level attributes.')
        for k,v in oCCDataSet.attrs.items():
            if verb>1: print('\t%s : %s ...'%(str(k),str(v)), end="")
            p = dict(v.__dict__)
            if 'type' in p:
                del p['type']   # for some reason value of type is an object which is not json serializable
            h5_CCDataSet.attrs[k] = json.dumps(p,default=tf.to_python_native)
            if verb>1: print('saved.')
        #

        if verb>0: print('Saving variable names.')
        if hasattr(oCCDataSet,'variable_names'):
            # If oCCDataSet.variable_names exists, we'll just dump them in here.
            varnames = np.array( oCCDataSet.variable_names, dtype=np.string_)

            # This is a little ugly. If there are variable descriptions,
            # store as a 2-by-n-array instead of an n-array. Loading will have
            # to handle the same thing.
            if len(oCCDataSet.variable_descrs)>0:
                vardescrs = np.array( oCCDataSet.variable_descrs, dtype=np.string_)
                blah = np.array([varnames,vardescrs])
            else:
                blah = varnames
            #

            h5_StudyAttrs = h5_CCDataSet.create_dataset('study_info', data=blah, compression="gzip", compression_opts=compression_level)
        else:
            h5_StudyAttrs = h5_CCDataSet.create_dataset('study_info', shape=(0,), compression="gzip", compression_opts=compression_level)
        #

        if verb>0: print('Saving study attributes.')
        # Dump the other dictionary into the attributes of the study_info h5dataset.
        for k,v in oCCDataSet.study_attrs.items():
            if verb>1: print('\t%s : %s ...'%(str(k),str(v)), end="")
            p = dict(v.__dict__)
            if 'type' in p:
                del p['type']   # for some reason value of type is an object which is not json serializable
            h5_StudyAttrs.attrs[k] = json.dumps(p,default=tf.to_python_native)
            if verb>1: print('saved.')
        #

        if verb>0: print('Saving feature sets in a subgroup (subfolder).')
        h5_FeatureSets = h5_CCDataSet.create_group('feature_sets')
        for k,v in oCCDataSet.feature_sets.items():
            if verb>1: print('\tCreating dataset %s ... '%str(k),end="")
            h5_FeatureSets.create_dataset(k, data=v, compression="gzip", compression_opts=compression_level)
            if verb>1: print('done.')
        #

        if verb>0: print('Saving subroupg with information about all metadata.')
        h5_metadata_table = h5_CCDataSet.create_group('metadata_table')
        h5_metadata_table.create_dataset('rows', data=meta_rows, compression="gzip", compression_opts=compression_level)
        h5_metadata_table.create_dataset('columns', data=np.array(meta_columns,dtype=np.string_), compression="gzip", compression_opts=compression_level)
        h5_metadata_table.create_dataset('metadata', data=np.array(meta_data, dtype=np.string_), compression="gzip", compression_opts=compression_level)

        if verb>0: print('Saving datapoints and their attributes.')
        for i,d in enumerate(oCCDataSet.data):
            if verb>1: print('\tSaving data point %i...'%i)

            # NOTE: We would like to handle the situation
            # when the datapoint is a generic collection of
            # lists (numpy arrays) that doesn't fit into a nice matrix.
            # Our solution here is to create a collection of HDF5 groups (subfolders)
            # for each element in the collection.
            # This seems to be the easiest fix in the existing framework.

            if d.dtype==np.dtype('O') or (not tf.has_homogeneous_elements(d)):
                # This is the key difference: this is a group instead
                # of a dataset now. It seems groups can have attributes
                # in the same way as datasets without any difficulties.
                h5_CCDataPoint = h5_CCDataSet.create_group(d._id)

                for i,elem in enumerate(d):
                    h5_CCDataPoint.create_dataset(str(i), data=elem, compression="gzip", compression_opts=compression_level)
                #
            else:
                h5_CCDataPoint = h5_CCDataSet.create_dataset(d._id, data=d, compression="gzip", compression_opts=compression_level)
            #

            for k in d.attrnames:
                # test comment
                # if k != 'attrnames' and k != '_id':

                #
                if k not in ['attrnames', '_id', '_loaded']:
                    if verb>1: print('\t\tSaving %s ...'%k,end="")
                    
                    p = dict(getattr(d,k).__dict__)
                    if 'type' in p:
                        del p['type']   # for some reason value of type is an object which is not json serializable
                    h5_CCDataPoint.attrs[k] = json.dumps(p,default=tf.to_python_native)
                    if verb>1: print('done.')
            #
            if verb>1: print('\tdone.')
        #
    #
    if verb>0: print('Dataset saved without errors.')
#

def load_CCDataSet(oCCDataSet, filename, **kwargs):
    '''
    Loads a CCDataSet from the given filename.
    Saved HDF file must have same version (set as attribute) as CCDataSet

    Inputs:
        oCCDataSet: CCDataSet object
        filename: String, indicating the location and name filename

    Outputs:
        CCDataSet: CCDataSet object (data loaded from HDF)

    Optional inputs:
        verbosity: specifying amount of printed output. (default: 0)
        debug: developer debug flag. if there's a pdb.set_trace() lying
            around, it will get triggered if this is True. (default: False)


        preload: Boolean. If True, entire dataset is loaded immediately.
            If False, the data itself isn't loaded; only
            the additional datasets listed above are loaded and
            placeholders are created for all the data, with capability to
            load data from the HDF file on the fly. (default: True)
    '''
    import h5py
    import numpy as np
    import json
    from calcom.io import CCDataPoint,CCList,CCDataAttr
    import copy


    verb = kwargs.get('verbosity',0)
    debug = kwargs.get('debug',False)
    preload = kwargs.get('preload',True)

    with h5py.File(filename, 'r') as f:
        if f.attrs['version'] != oCCDataSet.version:
            raise IOError("Version mismatch between CCDataSet and HDF file.")
        #
        h5_CCDataSet = f['CCDataSet']

        # containing all the ccdatapoints.
        dp_ccList = CCList()

        try:
            meta_rows = h5_CCDataSet['metadata_table']['rows'][()]
            meta_columns = h5_CCDataSet['metadata_table']['columns'][()]
            meta_columns = [c.decode('utf-8') for c in meta_columns]
            meta_data = h5_CCDataSet['metadata_table']['metadata'][()]

            oCCDataSet._metadata_table = (meta_rows,meta_columns,meta_data)
        except:
            if verb>0: print('Warning: loading metadata failed.')
            oCCDataSet._metadata_table = None
        #

        if not preload:
            # Create empty datapoints based on the metadata only.


            # NOTE: THIS IS A POOR MAN'S IMPLEMENTATION.
            # FURTHER DATA NEEDS TO BE SAVED IN METADATA_TABLE
            # INDICATING TYPES AND FORMATS OF THE VARIABLES
            # SO THAT THEY CAN BE DECODED PROPERLY.
            for i in meta_rows:
                oCCDataPoint = copy.deepcopy(CCDataPoint([]))
                oCCDataSet.ids.append(meta_data[i][0].decode('utf-8'))
                oCCDataPoint._id = meta_data[i][0].decode('utf-8')
                oCCDataSet.ids.append(oCCDataPoint._id)

                for j,e in enumerate(meta_data[i][1:]):
                    attrdictstr = '{'
                    attrdictstr += '"name": ' + '"%s", '%meta_columns[j+1]

                    try:
                        e_temp = int(e)
                        isnumeric = 'true' # THIS ISN'T ALWAYS TRUE!!!
                        attrdictstr += '"value": ' + '%i, '%e_temp
                    except:
                        try:
                            e_temp = float(e)
                            isnumeric = 'true'
                            attrdictstr += '"value": ' + '%f, '%e_temp
                        except:
                            e_temp = e.decode('utf-8')
                            isnumeric = 'false'
                            attrdictstr += '"value": ' + '"%s", '%e_temp
                        #
                    #

                    attrdictstr += '"is_numeric": ' + '%s, '%isnumeric
                    attrdictstr += '"is_derived": false, '
                    attrdictstr += '"_CCDataAttr__long_description": null'
                    attrdictstr += '}'

                    x = json.loads(attrdictstr)
                    attr = CCDataAttr()
                    attr.__dict__ = x
                    setattr(oCCDataPoint, x['name'], attr)
                    oCCDataPoint.attrnames.append(x['name'])
                #

                oCCDataPoint._loaded = False

                dp_ccList.append(oCCDataPoint)
            #
        #

        if verb>0: print('Loading the about/readme.')
        # h5_CCDataSet.attrs['about'] =  oCCDataSet._about_str
        try:
            # oCCDataSet._about_str = h5_CCDataSet.attrs['about']
            oCCDataSet._about_str = f.attrs['about']
            if verb>1: print('done.')
        except:
            # Should this be handled with an update to the version?
            # How do we handle old datasets?
            if verb>1: print('Loading about/readme FAILED.')
            oCCDataSet._about_str = ''
        #


        if verb>0: print('Loading attribute information for the dataset.')
        for k in h5_CCDataSet.attrs:
            x = json.loads(h5_CCDataSet.attrs[k])
            attr = CCDataAttr()
            attr.__dict__ = x
            oCCDataSet.attrs[k] = attr
            oCCDataSet.attrnames.append(x['name'])
        if verb>1: print('done.')

        # TODO: RESTRUCTURE THIS CODE SO THAT DIFFERENT COMPONENTS ARE
        # LOADED IN A CLEAN WAY RATHER THAN DOING A FOR-LOOP AND
        # CHECKS WITH IF-STATEMENTS.
        dp_ids = list(h5_CCDataSet)
        others = ['feature_sets', 'study_info', 'metadata_table']
        for o in others:
            try:
                dp_ids.remove(o)
            except:
                continue
        #

        ###################

        if verb>0: print('Loading study info.')
        # point = h5_CCDataSet['study_info']
        studyInfo = h5_CCDataSet['study_info']
        # import pdb
        # pdb.set_trace()
        temp = np.array( studyInfo[()], dtype=np.dtype('U') )
        if len(temp)>0:
            if len(np.shape(temp)) > 1: #Check for variable descrs
                oCCDataSet.variable_names = CCList(temp[0,:])
                oCCDataSet.variable_descrs = CCList(temp[1,:])
            else:
                oCCDataSet.variable_names = CCList(temp)
        #

        try:
            for k in studyInfo.attrs:
                x = json.loads(studyInfo.attrs[k])
                attr = CCDataAttr()
                attr.__dict__ = x
                oCCDataSet.study_attrs[k] = attr
                # oCCDataSet.attrnames.append(x['name'])
        except:
            pass
        #

        ###################

        if verb>0: print('Loading feature sets.')
        oCCDataSet.feature_sets = {}
        fsets = h5_CCDataSet['feature_sets']
        for fset in fsets:
            oCCDataSet.feature_sets[fset] = CCList(h5_CCDataSet['feature_sets'][fset])

        if not preload:
            if verb>0: print('Preloading disabled; exiting without loading datapoint values.')
            oCCDataSet.data = dp_ccList
            return oCCDataSet
        #

        ###################
        if verb>0: print('Loading datapoints.')
        for _id in dp_ids:
            # Have to put a switch in here.
            # Everything else should be a datapoint.
            point = h5_CCDataSet[_id]

            # if verb>0: print('Loading datapoint %s'%str(_id))
            # NOTE: for CCDataSets whose CCDataPoints are
            # not simple np.arrays, they are saved as
            # Groups in an HDF file.
            #
            # Thus, if a group, then we need to stitch together the
            # datapoint before moving forward.
            # if False:
            if type(point) == h5py._hl.group.Group:
                point_temp = CCList(range(len(point)))
                for j,(k,v) in enumerate( point.items() ):
                    point_temp[j] = v[()]
                #
                oCCDataPoint = copy.deepcopy(CCDataPoint(point_temp))
            else:
                # For some reason, attrnames gets passed between
                # multiple datapoints, ending up with many-times duplicated lists.
                # This is a fix but I don't know if there's a faster solution.
                # oCCDataPoint = CCDataPoint(point[:])
                # oCCDataPoint = copy.deepcopy(CCDataPoint(point[:]))
                oCCDataPoint = copy.deepcopy(CCDataPoint(point[()]))
            #

            oCCDataSet.ids.append(_id)

            for k in point.attrs:
                x = json.loads(point.attrs[k])
                attr = CCDataAttr()
                attr.__dict__ = x
                setattr(oCCDataPoint,x['name'], attr)
                oCCDataPoint.attrnames.append(x['name'])
            #

            oCCDataPoint._id = _id
            oCCDataPoint._loaded = True
            dp_ccList.append(oCCDataPoint)
            # #
        #
        oCCDataSet.data = dp_ccList
    #

    return oCCDataSet
#

def load_CCDataPoint(filename, identifier, **kwargs):
    '''
    Loads a single datapoint based on its identifier, which is
    common to the CCDataSet and the underlying HDF file.
    It is assumed the oCCDataPoint.id, at the very least, exists.
    Current implementation only loads the value of the data,
    assuming the attributes have already been loaded through
    ccd._metadata_table.

    Note that load_CCDataSet() is probably faster if you want to load
    everything; if you try to set up a large loop around this,
    it will open/close the HDF file every time.

    Inputs:
        identifier: string corresponding to the datapoint's attribute "_id" which
            is used to lookup the HDF Dataset (the CCDataPoint)

        filename: String; the filename on hard drive of the CCDataSet
            to look this up from.

    Optional inputs:
        verbosity: Integer; indicating amount of output. (Default: 0)
    Outputs:
        A CCDataPoint() corresponding to the identifier.
    '''

    # TODO - REPLACE THE PROCEDURE IN load_CCDataSet WITH THIS;
    # HAVE AN OPTIONAL/ALTERNATE INPUT ALLOWING FOR AN ALREADY OPENED
    # HDF FILE TO AVOID UNNECESSARY OPENING/CLOSING.
    import h5py
    import numpy as np
    import json
    from calcom.io import CCDataPoint,CCList,CCDataAttr
    import copy

    verb = kwargs.get('verbosity', 0)

    f = h5py.File(filename)

    id = identifier
    if verb>0: print('Loading data from _id %s...'%id)

    # I don't know of a good way of doing this other than overwriting
    # the datapoint.
    point = f['CCDataSet'][id]
    if type(point) == h5py._hl.group.Group:
        if verb>0: print('Datapoint has multiple data vectors; loading.')
        point_temp = CCList(range(len(point)))
        for k,v in point.items():
            if verb>1: print('\tLoading datapoint %s entry %s.'%(id,str(k)))
            point_temp[int(k)] = v[()]
        #
        oCCDataPoint = copy.deepcopy( CCDataPoint(point_temp) )
    else:
        # For some reason, attrnames gets passed between
        # multiple datapoints, ending up with many-times duplicated lists.
        # This is a fix but I don't know if there's a faster solution.
        # oCCDataPoint = CCDataPoint(point[:])
        if verb>1: print('Loading datapoint %s.'%id)
        oCCDataPoint = copy.deepcopy( CCDataPoint(point[()]) )
    #
    if verb>1:
        print('\tDatapoint:')
        print(oCCDataPoint)

    if verb>0: print('Loading datapoint %s attributes.'%id)
    for k in point.attrs:
        x = json.loads(point.attrs[k])
        attr = CCDataAttr()
        attr.__dict__ = x
        setattr(oCCDataPoint,x['name'], attr)
        oCCDataPoint.attrnames.append(x['name'])
        if verb>1: print('\tDatapoint %s attribute %s: %s.'%(id,x['name'],attr[()]))
    #

    oCCDataPoint._id = id

    return oCCDataPoint
#
