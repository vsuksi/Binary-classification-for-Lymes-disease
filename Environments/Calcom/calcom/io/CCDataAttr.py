class CCDataAttr:
    '''
    This containts a single attribute of a CCDataPoint. The attribute
    will have some extra information attached to it, hence the need for
    a devoted class.
    '''

    def __init__(self):
        self.name = None
        self.value = None
        self.is_derived = False
        # self.type = None         # the "type" attribute is used by numpy or hdf
        self.is_numeric = False
        self.long_description = None

        return

    # Intering/pooling duplicate `long_description` within the process.
    # Please note that most string are autometically interned by python in compile time; if the are avilable at compile time that is.
    # This is done for strings that may not be available in compile time
    # for example:
    #       x, y = "foo", "foo"
    #       x is y
    #       > True
    #
    #       But,
    #
    #       x, y = input(), input()
    #       > "foo"
    #       > "foo"
    #       x is y
    #       > False
    #
    @property
    def long_description(self):
        return self.__long_description

    @long_description.setter
    def long_description(self,long_description):
        import sys
        if long_description:
            # force intern in runtime. make sure that duplicate string don't take up extra memory space
            self.__long_description = sys.intern(long_description)
        else:
            self.__long_description = long_description

    def __str__(self):
        # Pretty print this.
        # attrdict = self.__dict__
        rs = '\n'
        w0 = 20
        rs += 'CCDataAttr %s\n%s\n\n'%(self.name[:w0],'-'*(w0+12))

        for a in ['value','is_numeric','is_derived','long_description']:
            val = str(getattr(self,a)) + ' '*w0
            rs += '\t%s : %s\n'%(a[:w0],val)
        #
        return rs
    #
    def __repr__(self):
        # return str(self.__dict__)
        return self.__str__()
#
