import calcom
import numpy as np 

def test(**kwargs):
    '''
    Test 018: Does torch have a solve command? What about lstsq? gels?
    '''
    try:
        import torch
    except:
        return False
    #

    subtests = []

    if torch.__version__>="1.2.0":
        subtests.append( hasattr(torch, 'lstsq') )
        subtests.append( hasattr(torch, 'solve') )
    else:
        subtests.append( hasattr(torch, 'gels') )
    #

    return all(subtests)
#
