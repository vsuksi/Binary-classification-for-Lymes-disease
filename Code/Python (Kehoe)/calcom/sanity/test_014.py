def test(**kwargs):
    '''
    Test 014: Does BSR get calculated in
        calcom.metrics.ConfusionMatrix()
        in an expected way under unusual situations?
        Only dealing with "simple" BSR here.
    '''
    import calcom
    import numpy as np

    subtests = []

    # First - an easy case. Same labels present in both true and
    # predicted.
    cf = calcom.metrics.ConfusionMatrix()
    true = [0,0,1,0,1]
    pred = [1,0,0,0,0]
    target_bsr = 0.5*( 2./3 + 0./2 )
    cf.evaluate(true,pred)
    subtests.append( cf.results['bsr'] == target_bsr )

    # Second - more than two classes; again, all predicted labels
    # lie within the collection of true labels.
    cf = calcom.metrics.ConfusionMatrix()
    true = ['a','b','c','c','c','c']
    pred = ['a','c','c','c','c','c']
    target_bsr = 1./3*( 1./1 + 0./1 + 4./4 )
    cf.evaluate(true,pred)
    subtests.append( cf.results['bsr'] == target_bsr )

    # Third - two classes, but the true labels are only in single class.
    cf = calcom.metrics.ConfusionMatrix()
    true = ['lion', 'lion', 'lion', 'lion']
    pred = ['tiger', 'lion', 'tiger', 'lion']
    target_bsr = 1*(2./4)
    cf.evaluate(true,pred)
    subtests.append( cf.results['bsr'] == target_bsr )

    # Fourth - three classes, but the true labels are only in two classes.
    cf = calcom.metrics.ConfusionMatrix()
    true = ['lion', 'lion', 'lion', 'lion', 'bear' ,'bear']
    pred = ['tiger', 'lion', 'tiger', 'lion', 'bear', 'tiger']
    target_bsr = 1./2*( 2./4 + 1./2 )
    cf.evaluate(true,pred)
    subtests.append( cf.results['bsr'] == target_bsr )

    return all(subtests)
#

if __name__=="__main__":
    print( test() )
