def test(**kwargs):
    '''
    Test 015: Can we successfully save to disk and load
        from disk a "simple" CCDataSet?
    '''
    import calcom
    import numpy as np
    import sys
    import os

    if kwargs.get('d',False):
        import pdb
        pdb.set_trace()
    #

    subtests = []

    ccd0 = calcom.utils.synthetic_datasets.generate_synthetic_ccd1()
    ccd1 = calcom.utils.synthetic_datasets.generate_synthetic_ccd2()

    #############################################
    #
    # Try saving each dataset to disk.
    #

    tmp_folder = os.getcwd() + '/_tmp/'
    if not os.path.exists(tmp_folder):
        try:
            os.mkdir(tmp_folder)
        except:
            if kwargs.get('v',0)>0:
                print('\tFailed to create temporary directory to save CCDataSets for testing; aborting.')
            return False
        #
    #

    try:
        ccd0.save('./_tmp/ccd0.h5')
        subtests.append( True )
    except:
        subtests.append( False )
        # catch the exception and print
        etype, evalue, etraceback = sys.exc_info()
        # eh i don't care that much yet
    #

    try:
        ccd1.save('./_tmp/ccd1.h5')
        subtests.append( True )
    except:
        subtests.append( False )
        # catch the exception and print
        etype, evalue, etraceback = sys.exc_info()
        # eh i don't care that much yet
    #


    #############################################
    #
    # Try loading each dataset from disk.
    #

    try:
        ccd0a = calcom.io.CCDataSet('./_tmp/ccd0.h5')
        subtests.append( True )
    except:
        subtests.append( False )
    try:
        ccd1a = calcom.io.CCDataSet('./_tmp/ccd1.h5')
        subtests.append( True )
    except:
        subtests.append( False )

    #############################################
    #
    # Test the new and old datasets for equality.
    #
    # For now, numerically compare dataset values
    # and use pandas to compare metadata.
    #

    def dat(dset):
        return np.array(dset.data)
    #
    def compare(dset1,dset2):
        try:
            order1 = dset1.lexsort(['_id'])
            order2 = dset2.lexsort(['_id'])

            data_equality = np.all( dat(dset1)[order1] == dat(dset2)[order2] )

            # super confusing - this function forces a sort;
            # don't need to use the ordering above for comparison.

            df1 = dset1.generate_metadata(sortby='_id')
            df2 = dset2.generate_metadata(sortby='_id')

            attrs = df1.columns

            mdat_equality = np.all( df1[attrs].values == df2[attrs].values )

            return (data_equality and mdat_equality)
        except:
            return False
    #

    subtests.append( True if compare(ccd0,ccd0a) else False )
    subtests.append( True if compare(ccd1,ccd1a) else False )

    # clean up
    os.remove('./_tmp/ccd0.h5')
    os.remove('./_tmp/ccd1.h5')
    os.rmdir('./_tmp/')

    return all(subtests)
#

if __name__=="__main__":
    print( test() )
