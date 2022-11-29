def test(**kwargs):
    '''
    Test 003: does the random classifier work?
        More importantly, does the AbstractClassifier do its job
        in managing labels?
    '''

    import calcom
    import numpy as np

    n,d = 18,7
    data = np.random.randn(n,d)
    labels = np.random.choice(['lion','tiger'], n)

    rc = calcom.classifiers.RandomClassifier()

    rc.fit( data[:n//2], labels[:n//2] )
    pred = rc.predict(data[n//2:])

    return np.all(  [p in ['lion','tiger'] for p in pred]  )
#

if __name__=="__main__":
    print( test() )
