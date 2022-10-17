def scipy_linprog(c,A,b,**kwargs):
    '''
    This is a wrapper utilizing scipy's linear program solver
    scipy.optimize.linprog(). The inputs to this function are
    set up to solve:

    max c'x
        s.t. Ax>=b, x>=0.

    Note that scipy.optimize.linprog() separately accepts
    equality constraints, while this does not. (Equality constraints
    can always be enforced with a pair of inequalities.)

    Also note that scipy.optimize.linprog() uses the convention Ax<=b;
    internally we negate A and b so that the convention with
    calcom.solvers.LPPrimalDualPy() is maintained.

    Inputs:

        c, np.ndarray of type float or double, with c.shape = (N,)
        A, np.ndarray of type float or double, with A.shape = (M,N)
        b, np.ndarray of type float or double, with b.shape = (M,)

    Optional inputs:
        verbosity: level of output (default: 0). Note that this is
            distinct from setting options['disp']=True.

        MAX_ITERS       (integer, default 1000).
            NOTE that this overrides options['maxiter'] if specified.


    The following optional inputs are passed directly to linprog (the
    descriptions are copied from the docstring from scipy version 1.1.0).
    NOTE: The "options" dictionary for linprog is filled based on the
    values specified above for "verbosity" and "MAX_ITERS" kwargs to
    maintain compatibility with LPPrimalDualPy.

        method : str, optional
            Type of solver.  :ref:`'simplex' <optimize.linprog-simplex>`
            and :ref:`'interior-point' <optimize.linprog-interior-point>`
            are supported.
        callback : callable, optional (simplex only)
            If a callback function is provide, it will be called within each
            iteration of the simplex algorithm. The callback must have the
            signature ``callback(xk, **kwargs)`` where ``xk`` is the current
            solution vector and ``kwargs`` is a dictionary containing the
            following::

                "tableau" : The current Simplex algorithm tableau
                "nit" : The current iteration.
                "pivot" : The pivot (row, column) used for the next iteration.
                "phase" : Whether the algorithm is in Phase 1 or Phase 2.
                "basis" : The indices of the columns of the basic variables.

        options : dict, optional
            A dictionary of solver options. All methods accept the following
            generic options:

                maxiter : int
                    Maximum number of iterations to perform.
                disp : bool
                    Set to True to print convergence messages.


    Outputs: depends on the value of output_flag:
        The minimizer x, an np.ndarray with x.shape = (N,)

    '''

#    raise AssertionError("This solver shouldn't be used for now.")

    import numpy as np
    from scipy.optimize import linprog

    MAX_ITERS = kwargs.get('MAX_ITERS',1000)
    verbosity = kwargs.get('verbosity',0)

    linprog_opts = {'maxiter':MAX_ITERS, 'disp':True if verbosity>0 else False}
    lp_method = kwargs.get('method', 'simplex')
    if lp_method=='interior-point':
        linprog_opts['lstsq'] = True
    #

    for k,v in kwargs.get('options',{}).items():
        linprog_opts[k] = v
    #

    # Does the user want column vectors? If so, keep it in mind, then
    # reshape things for linprog.

    # Note: scipy.optimize.linprog expects constraints Ax<=b;
    # our primal dual code expects Ax>=b.
    if np.shape(b==2) and np.shape(b)[1]==1:
        column_b = True
        b_ub = b.flatten()
    else:
        column_b = False
        b_ub = b
    #
    if np.shape(c==2) and np.shape(c)[1]==1:
        column_c = True
        c_in = c.flatten()
    else:
        column_c = False
        c_in = c
    #

    A_ub = A
    b_ub = b_ub
    c_in = -c_in

    result = linprog(
        c_in,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=None,
        b_eq=None,
        bounds=None,
        method=lp_method,
        callback=kwargs.get('callback', None),
        options=linprog_opts
    )

    if not result.success:
        # TODO: some kind of fault tolerance?
        raise AssertionError('scipy_linprog failed with message: %s'%str(result.message))
    #

    output = (result.x).reshape(-1,1) if column_b else result.x

    return output

#
