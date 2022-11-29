import calcom
import numpy as np 

def test(**kwargs):
    '''
    Test 017: Can we import torch? 
    '''
    try:
        import torch
        return True
    except:
        return False
    #
    return False
#
