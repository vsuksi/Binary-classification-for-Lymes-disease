def test(**kwargs):
    '''
    Test 001: can we import some of the other "mission-critical" packages?
    '''
    try:
        import numpy,scipy,sklearn,pandas,h5py
        return True
    except:
        raise ImportError('Test failed: a dependency failed to import.')
    #
    return False    # should never get here
#

if __name__=="__main__":
    print( test() )
