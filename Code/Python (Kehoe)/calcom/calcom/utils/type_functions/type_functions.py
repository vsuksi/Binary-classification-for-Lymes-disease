'''
A loose collection of functions for checking types and casting from
numpy to python standard types. Possibly more in the future.
'''

import numpy as np
from calcom.io import CCList, CCDataPoint

numeric_likes = [int,float,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,np.float16,np.float32,np.float64]
integer_likes = [int,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64]
list_likes = [list,np.ndarray, CCList([]).__class__, CCDataPoint([]).__class__]
string_likes = [str,np.str,np.str_]
bytes_likes = [bytes,np.bytes_]
bool_likes = [bool,np.bool,np.bool8,np.bool_]

def is_numeric(value):
    '''
    Tests if the input is one of a few types recognized
    to be integer or ``real-valued", in either built-in python or numpy.
    '''
    return (type(value) in numeric_likes)
#

def is_integer(value):
    '''
    Tests if the input is one of a few types recognized
    to be integer, in either built-in python or numpy.
    '''
    return (type(value) in integer_likes)
#

def is_list_like(value):
    '''
    A loose test to see if we can loop over the entries of the input,
    excluding things such as strings.
    '''
    return ( hasattr(value,'__iter__') and (type(value) in list_likes) )
#

def is_string_like(value):
    '''
    A loose test to check if the value is string-like.
    Currently only used to validate inputs for re (regular expressions)
    package.
    '''
    return (type(value) in string_likes)
#

def is_bytes_like(value):
    '''
    A loose test to check if the value is string-like.
    Currently only used to validate inputs for re (regular expressions)
    package.
    '''
    return (type(value) in bytes_likes)
#

def is_bool_like(value):
    '''
    A loose test to check if the value is boolean-like.
    '''
    return (type(value) in bool_likes)

def has_homogeneous_elements(value):
    '''
    Check if all the elements in value have the same shape as
    the first element. Returns false for non-list-likes.
    Does not care if there are type differences in the elements;
    only cares about sizes.

    has_homogeneous_elements([1,2,3,4]) -> True
    has_homogeneous_elements([[1,2], [3], [4]]) -> False
    has_homogeneous_elements([np.arange(5), np.arange(10)]) -> False
    has_homogeneous_elements([1,2,[3]]) -> Error due to np.shape throwing an error.

    NOT PERFECT - you could trick this thing with nested lists.
    '''
    # if not is_list_like(value):
    #     return False
    # else:
    sizes = np.array([np.shape(_t) for _t in value])
    return np.all( sizes == sizes[0] ) or (0 in np.shape(sizes))    # second case is with non-list elements.


def to_python_native(value):
    '''
    Checks the type of the input and returns
    the "correct" casting to a native python type.
    '''
    if is_list_like(value):
        return list(value)
    elif is_bytes_like(value): # This needs to be decoded, not casted.
        return value.decode('utf-8')
    elif is_string_like(value):
        return str(value)
    elif is_bool_like(value):
        return bool(value)
    elif is_integer(value):
        return int(value)
    elif is_numeric(value):
        return float(value)
    elif value==None:
        return None
    else:
        raise ValueError('The input of type %s is an unrecognized type for casting'%(str(type(value))))
        return None
    #
#
