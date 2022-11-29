def cvxopt_lp(c,A,b,**kwargs):
    '''
    This is a wrapper utilizing cvxopt's linear program solver
    cvxopt.solvers.lp(). The inputs to this function are
    set up to solve:

    max c'x
        s.t. Ax>=b, x>=0.

    Note that scipy.optimize.linprog() separately accepts
    equality constraints, while this does not. (Equality constraints
    can always be enforced with a pair of inequalities.)

    Also note that signs of the inputs are flipped to comply
    with conventions of inequalities, etc. used by cvxopt.solvers.lp.

    Inputs:

        c, np.ndarray of type float or double, with c.shape = (N,)
        A, np.ndarray of type float or double, with A.shape = (M,N)
        b, np.ndarray of type float or double, with b.shape = (M,)

    Optional inputs:
        verbosity: level of output (default: 0).

        MAX_ITERS       (integer, default 1000).
            NOTE that this overrides options['maxiter'] if specified.
        tol (float, default 0.001)
            
    Other optional arguments can be passed directly to cvxopt.solvers.lp
    with the kwarg "lp_options", which is a dictionary passed directly
    to the solver. See http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    for details. Inputs above will be overridden by options set below.
        'show_progress' : boolean   (default: False if verbosity==0 else True)
        'maxiters'      : integer   (default: MAX_ITERS)
        'abstol'        : float     (default: tol)
        'reltol'        : float     (default: tol)
        'feastol'       : float     (default: tol)
        'refinement'    : integer   (default: 0)
        

    Outputs: depends on the value of output_flag:
        The minimizer x, an np.ndarray with shape conforming to the convention of the inputs
        (if c is a column vector, so is x, and vice versa).

    '''

    import numpy as np
    try:
        import cvxopt
    except:
        raise ImportError('Use of cvxopt_lp requires installation of the cvxopt package.')
    #

    MAX_ITERS = kwargs.get('MAX_ITERS',1000)
    verbosity = kwargs.get('verbosity',0)
    tol = kwargs.get('tol',0.001)

    # See http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    lp_options = kwargs.get('lp_options',{})
    lp_options_default = {  'show_progress': False if verbosity==0 else True,
                            'maxiters': MAX_ITERS,
                            'abstol':tol,
                            'reltol':tol,
                            'feastol':tol,
                            'refinement':0
                            }
    for k,v in lp_options_default.items():
        if k not in lp_options.keys():
            lp_options[k] = v
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

    # cvxopt needs positivity for x specified explicitly.
    A_ub = np.vstack( (A_ub, -np.eye(A_ub.shape[1])) )
    b_ub = np.hstack( (b_ub, np.zeros(A_ub.shape[1])) )
    
    c_in = cvxopt.matrix(c_in)
    A_ub = cvxopt.matrix(A_ub)
    b_ub = cvxopt.matrix(b_ub)

    result = cvxopt.solvers.lp(
        c_in,
        A_ub,
        b_ub,
        options = lp_options
    )

    if result['status']!='optimal' and (verbosity>0):
        print('cvxopt.solvers.lp reports the non-optimal message:')
        print(result['status'])
        return result
    #

    output = np.array(result['x']) if column_b else np.array(result['x']).flatten()

    return output

#
