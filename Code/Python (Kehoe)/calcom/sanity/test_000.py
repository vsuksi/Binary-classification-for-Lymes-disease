def test(**kwargs):
    '''
    Test 000: can we import calcom?
    '''
    try:
        import calcom
        return True
    except:
        # Last-ditch effort: are we on racecar on Katrina, specifically?
        try:
            import sys
            sys.path.append('/data3/darpa/calcom/')
            import calcom
            return True
        except:
            raise ImportError('Test failed: calcom failed to import.')
    #
    return False    # should never get here
#

if __name__=="__main__":
    print( test() )
