if __name__ == "__main__":
    '''
    This is a slightly involved script built to time
    a few linear programming solvers on a synthetic
    problem.

    Note: this relies on an optional installation of cvxopt,
    which is not one of the dependencies installed
    automatically installed by pip; either do a
    "pip install cvxopt" or similar, or see http://cvxopt.org/install/ .

    '''
    def setup_LP_ball(nconstr=100,ndim=2):
        '''
        Sets up a linear program which defines a linear function
        is optimized on a unit ball, where the ball is approximated by the
        intersection of a large number of half-planes.

        Returns c,A,b in the standard form

            min c'x
            s.t. Ax <= b
                 x >= 0

        For now we just fix c = -np.ones((ndim,1)). The constraints
        are randomized.
        '''
        import numpy as np

        c = -1.*np.ones((ndim,1))

        A = np.zeros((nconstr,ndim))
        b = np.zeros((nconstr,1))

        for i in range(nconstr):

            # Get random point on the ball.
            z = np.random.randn(ndim)
            z = z / np.sqrt(np.linalg.norm(z))

            A[i,:] = -z
            b[i] = -np.dot(z,z)
        #

        return c,A,b
    #


    if __name__ == "__main__":
        from calcom.solvers import LPPrimalDualPy, scipy_linprog, cvxopt_lp
        import numpy as np
        import time

        m = 500
        n = 20

        c,A,b = setup_LP_ball(m,n)

        start = time.time()
        x = LPPrimalDualPy(c,A,b, MAX_ITERS=1000, tol=10**-8)
        stop = time.time()

        start2 = time.time()
        x2 = scipy_linprog(c,A,b, MAX_ITERS=1000, solver='simplex', options={'tol':10**-8})
        stop2 = time.time()

        start3 = time.time()
        x3 = cvxopt_lp(c,A,b, MAX_ITERS=1000, solver='simplex', options={'tol':10**-8})
        stop3 = time.time()

        resid = np.dot(A,x) - b
        x = x.flatten()
        resid = resid.flatten()

        resid2 = np.dot(A,x2) - b
        x2 = x2.flatten()
        resid2 = resid2.flatten()

        resid3 = np.dot(A,x3) - b
        x3 = x3.flatten()
        resid3 = resid3.flatten()

        print("%s\nLPPrimalDualPy\n%s"%('='*40, '='*40))
        print("\nExecution time (seconds): %.2f \n" % (stop-start,))

        residpos = all(resid>=0)
        print('\nresidual Ax-b>=0? ' + str(residpos))
        if not residpos:
            prop = float(len(np.where(resid<0)[0])) / len(resid)
            print('Proportion of negative terms: %.4e'%prop)
            print('Maximum violation: %.4e'%resid.min())
            print('')
        #

        xpos = all(x>=0)
        print('All values of x are nonnegative? ' + str(xpos)+'\n')
        if not xpos:
            prop = float(len(np.where(x<0)[0])) / len(x)
            print('Proportion of negative terms: %.4f'%prop)
            print('Maximum violation: %.4e'%x.min())
            print('')
        #
        print('objective function value: %.10f'%np.dot(c.T,x))

        print('='*40)
        print('='*40)

        print("%s\nscipy_linprog\n%s"%('='*40, '='*40))
        print("\nExecution time (seconds): %.2f \n" % (stop2-start2,))

        residpos2 = all(resid2>=0)
        print('\nresidual Ax-b>=0? ' + str(residpos2))
        if not residpos2:
            prop2 = float(len(np.where(resid2<0)[0])) / len(resid2)
            print('Proportion of negative terms: %.4f'%prop2)
            print('Maximum violation: %.4e'%resid2.min())
            print('')
        #

        xpos2 = all(x2>=0)
        print('All values of x are nonnegative? ' + str(xpos2)+'\n')
        if not xpos2:
            prop2 = float(len(np.where(x2<0)[0])) / len(x2)
            print('Proportion of negative terms: %.4e'%prop2)
            print('Maximum violation: %.4e'%x2.min())
            print('')
        #
        print('objective function value: %.10f'%np.dot(c.T,x2))

        print('='*40)
        print('='*40)

        print("%s\ncvxopt_lp\n%s"%('='*40, '='*40))
        print("\nExecution time (seconds): %.2f \n" % (stop3-start3,))

        residpos3 = all(resid3>=0)
        print('\nresidual Ax-b>=0? ' + str(residpos3))
        if not residpos3:
            prop3 = float(len(np.where(resid3<0)[0])) / len(resid3)
            print('Proportion of negative terms: %.4f'%prop3)
            print('Maximum violation: %.4e'%resid3.min())
            print('')
        #

        xpos3 = all(x3>=0)
        print('All values of x are nonnegative? ' + str(xpos3)+'\n')
        if not xpos3:
            prop3 = float(len(np.where(x3<0)[0])) / len(x3)
            print('Proportion of negative terms: %.4e'%prop3)
            print('Maximum violation: %.4e'%x3.min())
            print('')
        #
        print('objective function value: %.10f'%np.dot(c.T,x3))
