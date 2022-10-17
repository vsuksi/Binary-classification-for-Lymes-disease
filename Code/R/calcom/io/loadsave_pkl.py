'''
A couple simple scripts to load/save pickle objects, by default
also using gzip to compress the pickle files.
'''
def load_pkl(filename,**kwargs):
    '''
    Loads a thing from the given filename, assuming in a pickle format.
    By default, assumes gzip compression has been used (as is used by
    default in save_to_disk()).

    Inputs:
        filename: String, indicating the location and name filename

    Outputs:
        thing: Whatever was stored in the file.

    Optional inputs:
        use_gzip: Boolean, indicating whether the file used gzip compression (default: True)
    '''
    import pickle
    use_gzip = kwargs.get('use_gzip',True)

    if use_gzip:
        import gzip
        fn = gzip.open(filename,'rb')
        thing = pickle.load(fn)
        fn.close()
    else:
        fn = open(filename,'rb')
        thing = pickle.load(fn)
        fn.close()
    #
    return thing
#

def save_pkl(obj,filename,**kwargs):
    '''
    Saves a thing to the given filename in a pickle format.
    By default, gzip is used to compress the file further, if possible.

    Inputs:
        obj: Whatever your thing is.
        filename: String, indicating the location and name filename

    Optional inputs:
        use_gzip: Boolean, indicating whether to use compression (default: True)
    '''
    import pickle
    use_gzip = kwargs.get('use_gzip',True)

    if use_gzip:
        import gzip
        fn = gzip.open(filename,'wb')
        pickle.dump(obj,fn)
        fn.close()
    else:
        fn = open(filename,'wb')
        pickle.dump(obj,fn)
        fn.close()
    #
    return
#

if __name__=="__main__":
    # Testing script. Try loading a classifier, training it on some
    # fake data, saving it, reloading it, and verifying the models
    # in the two classes are the same.

    import calcom
    import numpy as np

    myssvm = calcom.classifiers.SSVMClassifier()

    n = 30                      # Number of data points
    m = 100                     # Number of features

    fname = './ssvm_test.pklz'  # Name of file

    x = np.random.randn(n,m)
    y = np.random.randint(0,2,n)    # n random integers from {0,1}

    myssvm.fit(x,y)

    # Save to file
    save_pkl(myssvm, fname)

    # Load
    myssvm2 = load_pkl(fname)

    # Check
    print("Checking if the saved & loaded class matches the original: ", end='')
    print( all( myssvm.results['weight'] == myssvm2.results['weight'] ) )

    # Cleanup
    rm_pkl = input("Remove pickle file? (Y/n):") or 'y' # Default to 'y' if nothing entered.

    if rm_pkl!='n':
        import os
        os.remove(fname)
        print("File deleted.")
    #
