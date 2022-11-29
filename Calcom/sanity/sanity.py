'''
Purpose: a small collection of sanity checks for calcom.
Not quite a full test suite, in the sense of unit testing,
but these tests must be passed before pushing a change.

This module is the "master script" which
imports each file in this folder and attempts to run their
test() method.

Additionally contains an "about()" function which
prints the docstring for every test in the folder.
'''

def collect_tests():
    '''
    Parses the same folder as this script, looks
    for .py suffix files, then trims them, to
    be imported with importlib in the test() script.
    '''
    import os, glob

    folder = os.path.dirname(__file__)
    files = glob.glob(folder+'/*.py')
    files.sort()

    # Names of things to import and test
    test_modules = [f.split('/')[-1].split('.')[0] for f in files]

    windows_switch = ('\\' in test_modules[0])
    for tm in test_modules:
        if 'sanity' in tm[-6:]:
            test_modules.remove(tm)
            break
    #

    if windows_switch:
        # trim the head of the file names.
        test_modules = [ tm.split('\\')[-1] for tm in test_modules ]
    #

    return test_modules
#

def about(which=''):
    '''
    collects the test modules, sequentially prints
    the docstring for each test() function.

    If the optional argument 'which' is specified,
    only prints the docstring for the requested module's test().
    '''
    import importlib

    if len(which)>0:
        module = importlib.import_module(which)
        ds = '\n'.join( (module.test.__doc__).split('\n')[1:] )

        print('Module %s:\n=============================='%str(which))
        print(ds)
    else:
        test_modules = collect_tests()
        for thing in test_modules:
            module = importlib.import_module(thing)
            print('Module %s:\n=============================='%str(thing))
            ds = '\n'.join( (module.test.__doc__).split('\n')[1:] )

            print(ds,end='\n\n')
    #
    return
#

def test(**kwargs):
    '''
    whoah dude - so meta
    '''
    import importlib

    test_modules = collect_tests()

    # string formatting junk
    print('')
    base = 'Running {}.test()'
    success = {False: 'FAILED', True: 'PASSED'}

    passes = []
    fails = []
    #
    for i,f in enumerate(test_modules):
        base_formatted = base.format(f).ljust(len(base) + 12) + ' ... '

        print(base_formatted, end='')
        try:
            module = importlib.import_module(f)
            test_passed = module.test(**kwargs)
        except:
            test_passed = False
        #

        if test_passed:
            passes.append(f)
        else:
            fails.append(f)
        #
        print( success[test_passed] )
    #

    print( '\nTests completed. {} of {} tests passed.\n'.format( len(passes), len(test_modules) ) )

    return (len(fails)==0)
#

if __name__=="__main__":
    result = test()
#
