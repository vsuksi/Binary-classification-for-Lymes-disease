def test(**kwargs):
    '''
    Test 002: do the basic submodules exist?
    '''

    submodules = [
                    'ccexperiment','classifiers','io','metrics',
                    'parameter_search','preprocessors','solvers','synthdata',
                    'utils','visualizers'
                ]

    import calcom

    exists = [s in calcom.__dict__ for s in submodules]
    return all(exists)

#

if __name__=="__main__":
    print( test() )
