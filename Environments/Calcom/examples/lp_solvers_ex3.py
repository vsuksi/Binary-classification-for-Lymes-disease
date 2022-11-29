if __name__ == "__main__":
    '''
    This is a cousin to the second example; here we're
    investigating the scaling of timings as the size
    of the linear program increases.
    '''

    def setup_LP_ball(nconstr=100,ndim=2):
        '''
        Sets up a linear program which defines a linear function
        is optimized on a unit ball, where the ball is approximated by the
        intersection of a large number of half-planes.

        Returns c,A,b in the standard form

            min np.dot(c.T, x)
            s.t. np.dot(A,x) >= b
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


    from calcom.solvers import LPPrimalDualPy
    import numpy as np
    import time


    ms = [300,600,1200,2400,4800,9600]
    n = 100 # Fix the number of dimensions for now. Unclear how this affects the solvers.
    timings = np.zeros((len(ms),2)) # timings over all values of m; pytorch vs numpy.

    for i,m in enumerate(ms):
        print("==========================================")
        print("START test with m=%i\n"%m)

        c,A,b = setup_LP_ball(m,n)

        ############################ RUN WITH PYTORCH ############################################

        start = time.time()
        x = LPPrimalDualPy(c,A,b, MAX_ITERS=300, use_cuda=True)
        stop = time.time()

        resid = np.dot(A,x) - b
        x = x.flatten()
        resid = resid.flatten()

        print("%s\nLPPrimalDualPy (with pytorch)\n%s"%('='*40, '='*40))
        print("\nExecution time (seconds): %.2f \n" % (stop-start,))

        residpos = all(resid>=0)
        print('\nresidual b-Ax>=0? ' + str(residpos))
        if not residpos:
            prop = float(len(np.where(resid<0)[0])) / len(resid)
            print('Proportion of negative terms: %.4f'%prop)
            print('')
        #

        xpos = all(x>=0)
        print('All values of x are nonnegative? ' + str(xpos)+'\n')
        if not xpos:
            prop = float(len(np.where(x<0)[0])) / len(x)
            print('Proportion of negative terms: %.4f \n'%prop)
        #
        print('objective function value: %.10f'%np.dot(c.T,x))
        pytorch_value = np.asscalar(np.dot(c.T,x))
        pytorch_time = stop-start

        timings[i,0] = pytorch_time

        ############################# RUN WITH NUMPY ############################################


        start = time.time()
        x = LPPrimalDualPy(c,A,b, MAX_ITERS=300)
        stop = time.time()

        resid = np.dot(A,x) - b
        x = x.flatten()
        resid = resid.flatten()

        print("%s\nLPPrimalDualPy (with numpy)\n%s"%('='*40, '='*40))
        print("\nExecution time (seconds): %.2f \n" % (stop-start,))

        residpos = all(resid>=0)
        print('\nresidual b-Ax>=0? ' + str(residpos))
        if not residpos:
            prop = float(len(np.where(resid<0)[0])) / len(resid)
            print('Proportion of negative terms: %.4f'%prop)
            print('')
        #

        xpos = all(x>=0)
        print('All values of x are nonnegative? ' + str(xpos)+'\n')
        if not xpos:
            prop = float(len(np.where(x<0)[0])) / len(x)
            print('Proportion of negative terms: %.4f \n'%prop)
        #
        print('objective function value: %.10f'%np.dot(c.T,x))
        numpy_value = np.asscalar(np.dot(c.T,x))
        numpy_time = stop-start

        timings[i,1] = numpy_time

        ###########################################################################

        print( "Accuracy deviation of the GPU version from the CPU version: %.4f" %((pytorch_value - numpy_value)/numpy_value))


        print( "Performance gain by the GPU version: %.4f" %(numpy_time/pytorch_time) )
    #
