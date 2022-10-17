class CCList(list):
    '''
    A subclassed list, just so we can have a cleaner __repr__ and __str__.
    Possibly more functionality later (printing in a bit more detail what's there?)
    '''
    def __init__(self,*args):
        # Try to check for iterable input; in this case,
        # pass the args so that it just converts that type into this one.
        if hasattr(args,'__iter__'):
            list.__init__(self,*args)
        else:
            list.__init__(self,args)
        #
    #
    def __str__(self):
        ln = len(self)
        if ln==0:
            return "<Empty CCList>"
        else:
            s = ("<CCList with %i element(s) ["%ln)
            max_elems = 8
            # if less than `max_elems` elements in the list print everything
            if ln <= max_elems:
                for i in range(ln):
                    if i == 0:
                        s += str(self[i])
                    else:
                        s += ',' + str(self[i])
            # otherwise print from head and tail
            elif ln > max_elems:
                max_tail_len = 2
                for i in range(max_tail_len):
                        s += str(self[i]) + ','
                s += '...'
                for i in reversed(range(1,max_tail_len + 1)):
                    if i == 0:
                        s += str(self[i*-1])
                    else:
                        s += ',' + str(self[i*-1])
            s += ']>'
            return s
        #
    #
    def __repr__(self):
        return str(self)  # same as __str__
    #
    def __getitem__(self,val):
        '''
        Rudimentary slicing using pointers.
        '''
        from numpy import array,ndarray
        from calcom.utils import type_functions as tf
        # if type(val) in [slice] + tf.integer_likes:
        if isinstance(val, slice) or isinstance(val, int) or type(val) in tf.integer_likes:
            # we can do simple list slicing
            return super().__getitem__(val)
        elif type(val)==ndarray or isinstance(val,list):
            # Slice the thing ourselves; return a CCList.
            result = CCList([])
            for ii in val:
                result.append(super().__getitem__(ii))
            return result
        else:
            raise ValueError('Unrecognized slicing type %s.'%str(val))
        #
    #
#
