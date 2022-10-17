'''
NOTE: putting this function in utils "purgatory" 
for now, until the need for it arises again. 
Currently the way it's going, people are just using 
their own one-off code to do this.

1 October 2018
Manuchehr Aminian
'''

    def generate_relaxed_attr(self,attrname,bins=4,**kwargs):
        '''
        Given an attribute name, scour the entire CCDataSet on that variable,
        generating a new "relaxed" attribute. The attribute must be of a
        numerical type. The new attribute is generated from the minimum to the
        maximum value of the original attribute, with a specified number of
        bins.

        Inputs:
            attrname: String indicating the attribute name to apply this one.
            bins: Several options:
                * If bins=n (integer), n equally spaced bins are used from the
                  minimum to the maximum
                * If bins is a list of numerical values, the values are used
                  as the left and right endpoints of non-overlapping bins.
                * (Not Implemented Yet) If bins is a list of list of numeric values, the values are
                  used as intervals of over-lapping/non-overlapping bins.

        Optional inputs:
            idx_list: Integer list of indexes. If specified, only the given
                subset of the data is used for windowing. Defaults to
                np.arange(0,len(ccd.data)) (i.e., the whole dataset)

            overlap: (Not Implemented Yet) Boolean on whether to overlap bins
                if bins is an integer.  Defaults to False. Ignored if bins is
                any other type.
                If the bins are overlapped, then the derived labels are
                list-valued, and classification needs to be done based on set
                membership; looking like "j in labellist" rather than
                "j==label".
                NOTE: As of 12 Jan 2018, this still needs to be implemented.

            derived_attrname: String indicating the name of the new attribute.
                Defaults to the name of the original appended with a string
                indicating the width of the first bin.

        Outputs:
            None. However, a dictionary is created, stored under
            getattr(self,derived_attrname), which maps the old labels to the
            new labels.

        For example, if we have a calcom dataset ccd with datapoints with
        attribute 'time' given as:

            dp0['time'].value = 4
            dp0['time'].value = 1
            dp0['time'].value = 3
            dp0['time'].value = 6

        Then calling ccd.generate_windowed_attr('time',bins=2) creates a new
        attribute 'time_2bins' for all datapoints with bin intervals [1,3.5)
        and [3.5,6], and new labels

            dp0['time_2bins'].value = 1
            dp0['time_2bins'].value = 0
            dp0['time_2bins'].value = 0
            dp0['time_2bins'].value = 1

        '''
        FutureWarning('This function is under review and may be removed in the future.')
        import numpy as np
        from calcom.io import CCDataPoint
        from calcom.io import CCList
        from calcom.io import CCDataAttr

        from calcom.utils import type_functions as tf

        n = len(self.data)
        if tf.is_numeric(bins):
            bins = np.int64(bins)
            bin_size = bins
        elif tf.is_list_like(bins):
            bin_size = len(bins)
        #

        if not self.attrs[attrname].is_numeric:
            raise ValueError("Error: this function only supports attributes which take on numerical values.")
        #

        idx_list = kwargs.get('idx_list', np.arange(0,n, dtype=np.int64))
        overlap = kwargs.get('overlap',False)
        derived_attrname = kwargs.get('derived_attrname', attrname+'_'+str(bin_size)+'bins')

        attrvals = self.get_attr_values(attrname,**kwargs) # no need to pass idx_list explicitly

        minv = min(attrvals)
        maxv = max(attrvals)

        # If bins is list of lists of numeric values i.e. [[1,3],[3,7],[7,11]]
        if tf.is_list_like(bins) and len(bins)>0 and tf.is_list_like(bins[0]):
            intervals = bins
        # If bins is list of numerics i.e. [1,3,7,11]
        elif tf.is_list_like(bins):
            if not np.all(np.sort(bins) == bins):
                raise ValueError("bins must be monotonically increasing or decreasing")

            binned_attrvals = np.digitize(attrvals,bins);
            self.append_attr(derived_attrname, binned_attrvals,**kwargs)
            return binned_attrvals
        else:
            if not overlap:
                boundaries = np.linspace(minv,maxv,bins+1)
                boundaries[-1] += 1. # Hacky; it's to avoid an edge case for last interval.

                binned_attrvals = np.digitize(attrvals,boundaries);
                self.append_attr(derived_attrname, binned_attrvals,**kwargs)
                return binned_attrvals
        #
    #
