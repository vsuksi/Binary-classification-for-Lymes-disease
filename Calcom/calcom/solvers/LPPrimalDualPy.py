def LPPrimalDualPy(c,A,b,**kwargs):
    '''
    This is a Python implementation of a code solving
    a linear program of the form:

    max c'x
        s.t. Ax>=b, x>=0.

    Note this may be a different ``standard form" than seen elsewhere.
    Original version of this in calcom was named ``rKKTxy" and lived in
    ssvmclassifier.py. Before that, it was a Matlab code by
    Sofya Chepushtanova and Michael Kirby, and later implemented in Python by
    Tomojit Ghosh.

    Inputs:

        c, np.ndarray of type float or double, with c.shape = (N,1)
        A, np.ndarray of type float or double, with A.shape = (M,N)
        b, np.ndarray of type float or double, with b.shape = (M,1)

    Optional inputs:

        output_flag     (integer, default 0, indicating form of the output)
        verbosity       (integer, default 0. Nonzero => print statements used)
        use_cuda        (boolean, default False. indicating whether to use CUDA or not)

        max_iters       (integer, default 200)
        delta           (float or double, default 0.1)
        tol             (float or double, default 10**-3)
        dtheta_min      (float or double, default 10**-9)

        debug           (boolean, default False. If True, immediately turns on the Python debugger.)

    Outputs: depends on the value of output_flag:

        output_flag==0: (default)
            The minimizer x, an np.ndarray with x.shape = (N,)
        output_flag==1:
            IP, a dictionary which has detailed information about the
            solver and solution. This is used by
            calcom.classifiers.SSVMClassifier(), as it has auxilliary
            information used in solving the sparse SVM optimization
            problem specifically.

            NOTE: this may change in the future to make this code less beholden
                to SSVM.

    '''
    if kwargs.get('debug', False):
        import pdb
        pdb.set_trace()
    #

    import numpy as np
    try:
        import torch
    except ImportError:
        torch = None

    max_iters = kwargs.get('max_iters',200)
    delta = kwargs.get('delta',0.1)
    tol = kwargs.get('tol',0.001)

    dtheta_min = kwargs.get('dtheta_min',np.double(10.)**-9)
    output_flag = kwargs.get('output_flag',0)
    verbosity = kwargs.get('verbosity',0)

    use_cuda = kwargs.get('use_cuda',False) # set to True for large enough problem sizes

    N=len(c)
    M=len(b)


    # TODO: (check)
    # Is there a reason these are initialized in this way instead of randn?
    x,z,e3=np.ones(N).reshape(-1,1), np.ones(N).reshape(-1,1), np.ones(N).reshape(-1,1)
    p,w,e2=np.ones(M).reshape(-1,1), np.ones(M).reshape(-1,1), np.ones(M).reshape(-1,1)
    E0=np.vstack((x,w,p,z)) #initial point
    theta=np.zeros(max_iters)

    err=999 # Placeholder initial error.

    # Copy data into GPU
    if torch and torch.cuda.is_available() and use_cuda:
        A_c = torch.from_numpy(A).double().cuda()
        b_c = torch.from_numpy(b).double().cuda()
        c_c = torch.from_numpy(c).double().cuda()
        e2_c = torch.from_numpy(e2).double().cuda()
        e3_c = torch.from_numpy(e3).double().cuda()
        x_c = torch.from_numpy(x).double().cuda()
        p_c = torch.from_numpy(p).double().cuda()
        w_c = torch.from_numpy(w).double().cuda()
        z_c = torch.from_numpy(z).double().cuda()
        theta_c = torch.from_numpy(theta).double().cuda()


    # Two versions of the code, depending on output_flag. A lot of the
    # iteration saving can be thrown out if we only care about the solution.
    if (output_flag==1):
        IP={'xx':[],'ww':[],'pp':[],'zz':[],'met1':[],'met2':[],'met3':[],'met4':[],'amet1':[],'amet2':[],'amet3':[],'amet4':[],'bmet1':[],'bmet2':[],'bmet3':[],'bmet4':[],'err':err,'exitflag':False,'itrs':0,'val':-999}
        for i in range(max_iters):

            if (verbosity): # May want a second verbosity level for this.
                print('Iteration:',i+1)
            #

            b1=b-np.dot(A,x)
            b2=c-np.dot(A.T,p)
            rho=-w+b1
            sig=z+b2
            gamma=np.asscalar(np.dot(z.T,x)+np.dot(p.T,w))
            mu=(delta*gamma)/(M+N)
            a1=x*z
            a2=p*w
            F=np.vstack((rho,sig,a1-(mu*e3),a2-(mu*e2)))
            a3=1.0/p
            a4=1.0/x
            a5=1.0/z
            nn1=(x*a5)
            t1=b2+(mu*a4)
            WW=np.repeat(nn1.T,len(A),0)
            AXZI2=A*WW
            r=b1-(mu*a3)-np.dot(A,(nn1*t1))
            R=-np.diag((a3*w)[:,0])-np.dot(AXZI2,A.T)
            #dp=np.linalg.solve(R,r)
            rvec = r    # r isn't a good variable name for debugging with pdb.

            #dp,_,_,_ = np.linalg.lstsq(R,rvec, rcond=None)

            numpy_version = '.'.join( np.__version__.split('.')[:2] )
            if numpy_version < '1.14':
                dp, _, _, _ = np.linalg.lstsq(R,rvec)
            else:
                dp, _, _, _ = np.linalg.lstsq(R,rvec, rcond=None)
            #


            numpy_version = '.'.join( np.__version__.split('.')[:2] )
            if numpy_version < '1.14':
                dp, _, _, _ = np.linalg.lstsq(R,rvec)
            else:
                dp, _, _, _ = np.linalg.lstsq(R,rvec, rcond=None)
            #


            #dx=np.multiply(np.diag((x*a5)[:,0]).reshape(-1),(t1-np.dot(A.T,dp)).reshape(-1)).reshape(-1, 1)
            dx=np.dot(np.diag((x*a5)[:,0]),(t1-np.dot(A.T,dp)))
            dz=(mu*a4)-z-(a4*z)*dx
            dw=(mu*a3)-w-(a3*w)*dp
            mm=np.max(np.vstack((-dx/x,-dp/p,-dz/z,-dw/w)))
            newtheta=0.9/mm     # Is this something that should be made a parameter?
            theta[i]=newtheta

            if theta[i]<dtheta_min:   #if the step size is too small stop
                if (verbosity):
                    print('Step size is too small: ',theta[i])
                #
                break
            #
            x=x+theta[i]*dx
            p=p+theta[i]*dp
            z=z+theta[i]*dz
            w=w+theta[i]*dw

            IP['xx'].append(x)
            IP['pp'].append(p)
            IP['zz'].append(z)
            IP['ww'].append(w)
            IP['met1'].append(np.linalg.norm(rho))          #measure of primal constraint
            IP['met2'].append(np.linalg.norm(sig))          #measure of dual constraint
            IP['met3'].append(gamma)                        #measure of complementarity
            IP['met4'].append(np.linalg.norm(F))            #value of F that should be zero for a solution
            IP['amet1'].append(np.linalg.norm(rho,1))       #measure of primal constraint
            IP['amet2'].append(np.linalg.norm(sig,1))       #measure of dual constraint    IP['amet3'].append(np.linalg.norm(gamma,1))#measure of complementarity
            IP['amet3'].append(gamma)                       #measure of complementarity
            IP['amet4'].append(np.linalg.norm(F,1))         #value of F that should be zero for a solution
            IP['bmet1'].append(np.linalg.norm(rho,np.inf))  #measure of primal constraint
            IP['bmet2'].append(np.linalg.norm(sig,np.inf))  #measure of dual constraintIP['bmet3'].append(np.linalg.norm(gamma,np.inf))#measure of complementarity
            IP['bmet3'].append(gamma)                       #measure of complementarity
            IP['bmet4'].append(np.linalg.norm(F,np.inf))    #value of F that should be zero for a solution

            # These two don't seem to be used anywhere.
            # errb=np.max([np.max(IP['bmet1'][-1]),np.max(IP['bmet2'][-1]),np.max(IP['bmet3'][-1]),np.max(IP['bmet4'][-1])])
            # erra=np.max([np.max(IP['amet1'][-1]),np.max(IP['amet2'][-1]),np.max(IP['amet3'][-1]),np.max(IP['amet4'][-1])])
            err=np.max([np.max(IP['met1'][-1]),np.max(IP['met2'][-1]),np.max(IP['met3'][-1]),np.max(IP['met4'][-1])])

            if (verbosity): #May want a second verbosity level for this.
                print('Error:',err)
            #

            if err<tol:
                IP['val']=np.dot(c.T,x)
                IP['itrs']=i
                IP['exitflag']=True
                IP['err']=err
                break
            else:
                IP['itrs']=i
                IP['err']=err
            #
        #

        IP['x']=x
        IP['w']=w
        IP['p']=p
        IP['z']=z
        IP['conerr']=np.dot(A,x)-b

        # IP['weight']=x[:inputDim]-x[inputDim:2*inputDim]
        # IP['gamma']=x[2*inputDim]-x[2*inputDim+1]
    elif (output_flag==0):
        #CUDA / GPU Version
        if torch and torch.cuda.is_available() and use_cuda:

            for i in range(max_iters):

                if (verbosity): # May want a second verbosity level for this.
                    print('Iteration:',i+1)
                b1_c = torch.addmm(b_c, A_c , x_c, alpha = -1)
                b2_c = torch.addmm(c_c, A_c.t(), p_c, alpha = -1)
                rho_c = torch.add(b1_c, -1, w_c)
                sig_c = torch.add(b2_c, z_c)
                gamma = (torch.mm(z_c.t(), x_c) + torch.mm(p_c.t(), w_c))[0]
                mu = (delta * gamma) / (M + N)
                a1_c = x_c * z_c
                a2_c = p_c * w_c
                F = torch.cat((rho_c, sig_c, a1_c - (mu * e3_c), a2_c - (mu * e2_c)))
                a3_c = p_c.reciprocal()
                a4_c = x_c.reciprocal()
                a5_c = z_c.reciprocal()
                nn1_c = (x_c * a5_c)
                t1_c = b2_c + (mu * a4_c)
                WW_c = nn1_c.t().repeat(len(A),1)
                AXZI2_c = A_c * WW_c
                r_c = b1_c - (mu * a3_c) - torch.mm(A_c, (nn1_c * t1_c))
                # this line is causing the most significant difference
                # in the outputs of numpy and pytorch versions
                R_c = - torch.diag((a3_c* w_c)[:, 0]) - torch.mm(AXZI2_c, A_c.t())
                #import pdb;pdb.set_trace()
                # This will throw an exception if there are linear dependencies in the data
                dp_c,_ = torch.solve(r_c.view(-1,1),R_c)
                #
                dx_c = ((x_c * a5_c).view(-1) * (t1_c - torch.mm(A_c.t(), dp_c)).view(-1)).view(-1, 1)

                dz_c = (mu * a4_c) - z_c - (a4_c * z_c) * dx_c
                dw_c = (mu * a3_c) - w_c - (a3_c * w_c) * dp_c
                mm = torch.max( torch.cat( (- dx_c / x_c, - dp_c / p_c, -dz_c / z_c, -dw_c / w_c)))
                newtheta = 0.9 / mm
                theta_c[i] = newtheta
                if theta_c[i] < dtheta_min:   #if the step size is too small stop
                    if (verbosity):
                        print('Step size is too small: ',theta_c[i])
                    #
                    break
                #

                x_c = x_c + theta_c[i] * dx_c
                p_c = p_c + theta_c[i] * dp_c
                z_c = z_c + theta_c[i] * dz_c
                w_c = w_c + theta_c[i] * dw_c

                met1 = torch.norm(rho_c)
                met2 = torch.norm(sig_c)
                met3 = gamma
                met4 = torch.norm(F)

                err = np.max([met1, met2, met3, met4])

                if (verbosity): # May want a second verbosity level for this.
                    print('Error:',err)
                #

                if err<tol:
                    break
                #

        else:
            # Numpy Version
            for i in range(max_iters):

                if (verbosity): # May want a second verbosity level for this.
                    print('Iteration:',i+1)
                #

                b1 = b - np.dot(A, x)
                b2 = c - np.dot(A.T, p)
                rho = -w + b1
                sig = z + b2
                gamma = np.asscalar(np.dot(z.T, x) + np.dot(p.T, w))
                mu = (delta * gamma) / (M + N)
                a1 = x * z
                a2 = p * w
                F = np.vstack((rho, sig, a1 - (mu * e3), a2 - (mu * e2)))
                a3 = 1.0 / p
                a4 = 1.0 / x
                a5 = 1.0 / z
                nn1 = (x * a5)
                t1 = b2 + (mu * a4)
                WW = np.repeat(nn1.T, len(A), 0)
                AXZI2 = A * WW
                r = b1 - (mu * a3) - np.dot(A, (nn1 * t1))
                R = - np.diag((a3 * w)[:, 0]) - np.dot(AXZI2, A.T)
                #dp=np.linalg.solve(R,r)
                rvec=r  # "r" isn't a good variable name for debugging with pdb.
#                import pdb
#                pdb.set_trace()

                #dp,_,_,_=np.linalg.lstsq(R,rvec, rcond=None)
                numpy_version = '.'.join( np.__version__.split('.')[:2] )
                if numpy_version < '1.14':
                    dp, _, _, _ = np.linalg.lstsq(R,rvec)
                else:
                    dp, _, _, _ = np.linalg.lstsq(R,rvec, rcond=None)
                #

                dx = np.dot( np.diag( (x * a5)[:, 0]), (t1 - np.dot(A.T, dp)))
                dz = (mu * a4) - z - (a4 * z) * dx
                dw =( mu * a3) - w - (a3 * w) * dp
                mm = np.max( np.vstack( (-dx / x, -dp / p,  -dz / z, -dw / w)))
                newtheta = 0.9 / mm     # Is this something that should be made a parameter?
                theta[i] = newtheta
                if theta[i] < dtheta_min:   #if the step size is too small stop
                    if (verbosity):
                        print('Step size is too small: ',theta[i])
                    #
                    break
                #

                x = x + theta[i] * dx
                p = p + theta[i] * dp
                z = z + theta[i] * dz
                w = w + theta[i] * dw

                met1 = np.linalg.norm(rho)
                met2 = np.linalg.norm(sig)
                met3 = gamma
                met4 = np.linalg.norm(F)

                err=np.max([met1, met2, met3, met4])

                if (verbosity): # May want a second verbosity level for this.
                    print('Error:',err)
                #

                if err<tol:
                    break
                #


        #
    #

    # Copy necessary data from GPU to host memory
    if torch and torch.cuda.is_available() and use_cuda:
        x = x_c.cpu().numpy()
        theta = theta_c.cpu().numpy()

    if (output_flag==1):
        return IP
    elif (output_flag==0):
        return x
    else:
        return None
    #
#
