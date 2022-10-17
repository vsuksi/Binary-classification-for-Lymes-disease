from numpy import ndarray,asarray

class CCDataPoint(ndarray):
    '''
    A single "CCDataPoint" is a numpy.ndarray with extra attributes attached.
    In practice, the object is automatically populated with CCDataAttr(s),
    which is assumed an iterable of strings (that is, a list or similar of strings).

    You shouldn't typically be instantiating this on its own; use the functions in CCDataSet.

    Basic mechanism for subclassing a numpy.ndarray correctly is copied from
    https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

    '''

    def __new__(cls, input_array, attrnames=[]):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.attrnames = attrnames
        # Finally, we must return the newly created object:
        return obj
    #

    # The purpose of this is still pretty unclear for me.
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.attrnames = getattr(obj, 'attrs', [])
    #

    # This is just to have a one-liner to set all the attributes at once.
    # Note that the expected "attrs" input is a list of CCDataAttr(s),
    # NOT a list of strings! The idea being that the default values for the
    # the other object's attributes are defined in the CCDataSet, and
    # just passed here. Only the 'value' entries are updated.
    # def set_attrs(self,attrs,newattrs,newvalues):
    def set_attrs(self,newattrs,newvalues):
        from calcom.io import CCList
        try:
            attrnames = CCList(self.__dict__.keys())
        except:
            attrnames = CCList()
        #

        for i,newattr in enumerate(newattrs):

            if newattr.name not in attrnames:
                setattr(self,newattr.name,newattr)
            #
            ccattr = getattr(self,newattr.name)
            setattr(ccattr,'value',newvalues[i])
        #

        setattr(self,'attrnames',CCList(self.__dict__.keys()))
        return
    #

    def get_attr(self, attr_name):
        return getattr(self, attr_name, None)
    #

    # A quick and dirty way to visualize the attributes of a data point
    def get_all_attr_string(self, formatter='\t'):
        from calcom.io import CCDataAttr
        str1 = ""
        anames = list(self.attrnames)
        anames.sort()
        for attrname in anames:
            if isinstance(self.get_attr(attrname), CCDataAttr):
                str1 += attrname + ":" + str(self.get_attr(attrname).value) + ","+formatter
        return str1
    #

#
