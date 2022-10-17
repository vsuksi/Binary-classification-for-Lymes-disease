### ScaledConjugateGradient.py
### by Chuck Anderson, for CS545 and CS645
### http://www.cs.colostate.edu/~anderson/cs645
### You may use, but please credit the source.

#from copy import copy
import numpy as np
import sys
#from math import sqrt, ceil
#from collections import deque
import pdb

floatPrecision = sys.float_info.epsilon

######################################################################
### Scaled Conjugate Gradient algorithm from
###  "A Scaled Conjugate Gradient Algorithm for Fast Supervised Learning"
###  by Martin F. Moller
###  Neural Networks, vol. 6, pp. 525-533, 1993
###
###  Adapted by Chuck Anderson from the Matlab implementation by Nabney
###   as part of the netlab library.
###
###  Call as   scg()  to see example use.


def scg(x, f,gradf, calcTrErr, calcValErr,*fargs, **params):
    """scg:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    tmpW = []
    trainingMSE = []
    validationMSE = []
    allW=[]

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.
    #pdb.set_trace()
    while j <= nIterations:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            '''
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,'ftrace':ftrace[:j] if ftracep else None,
                        'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None,
                        'reason':"limit on machine precision"}
            '''
            if kappa==0:
                print('Terminating as kappa is zero. Can\'t proceed further.')
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,'ftrace':ftrace[:j] if ftracep else None,
                        'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and j % max(1,np.ceil(nIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,
                        'weights':allW,'bestItr':None,'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,
                        'weights':allW,'bestItr':None,'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    print('Terminating as gradient is zero.')
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None,
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        tmpW.append(np.linalg.norm(x))
        allW.append(x)
        trainingMSE.append(calcTrErr(x, *fargs))
        validationMSE.append(calcValErr(x, *fargs))
        #pdb.set_trace()
        if verbose and (j % np.ceil(nIterations/4) == 0):
            print('After iteration ',j,' training error(MSE/CE): ',trainingMSE[-1],' Validation error(MSE/CE): ',validationMSE[-1])
        j += 1

    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],'weightNorm':tmpW,
            'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None,'reason':"did not converge"}

#def adam(W,fName,stepSize=0.005,beta1=0.9,beta2=0.999,epsilon=1e-8,nIterations=100,*args):
def adam(W,returnCost,returnGrad,calcTrErr,calcValErr,*fargs,**params):
    #pdb.set_trace()
    nIterations = params.pop("nIterations",100)
    stepSize = params.pop("stepSize",0.005)
    beta1 = params.pop("beta1",0.9)
    beta2 = params.pop("beta2",0.999)
    epsilon = params.pop("epsilon",1e-8)
    fTrace = []
    xTrace = []
    tmpW = []
    trainingMSE = []
    validationMSE = []
    allW = []
    m_t = np.zeros([len(W),1])
    v_t = np.zeros([len(W),1])
    W = W.reshape(-1,1)
    for itr in range(1,nIterations+1,1):
        #pdb.set_trace()
        f = returnCost(W,*fargs)
        df = returnGrad(W,*fargs).reshape(-1,1)
        #f,df=fName(W,*args)#Get the initial gradient and cost value of the objective function
        #df = df.reshape(-1,1)
        xTrace = np.append(xTrace,W)
        fTrace = np.append(fTrace,f)
        m_t = beta1*m_t+(1-beta1)*df #Calculate biased first moment estimate
        v_t = beta2*v_t+(1-beta2)*df**2 #Calculate biased second raw moment estimate
        m_cap = m_t/(1-beta1**itr) #Bias-corrected first moment
        v_cap = v_t/(1-beta2**itr) #Bias-corrected second moment
        W = W - stepSize*m_cap/(np.sqrt(v_cap)+epsilon) #Update parameter
        allW.append(W)
        trainingMSE.append(calcTrErr(W, *fargs))
        validationMSE.append(calcValErr(W, *fargs))
        if itr % np.ceil(nIterations/5) == 0:
            print('After iteration ',itr,' training error(MSE/CE): ',trainingMSE[-1],' Validation error(MSE/CE): ',validationMSE[-1])

    return {'x':W[:,0], 'f':fTrace[-1], 'nIterations':itr, 'xtrace':xTrace, 'ftrace':fTrace,'trMSE':trainingMSE,'valMSE':validationMSE,
    'weights':allW,'bestItr':None}

def scgWithDynamicTrData(x,f,gradf,createDynamicData,calcTrErr,calcValErr,*fargs, **params):
    """scg:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1						# j counts number of iterations.
    tmpW = []
    trainingMSE = []
    validationMSE = []
    allW=[]

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.
    while j <= nIterations:
        if j>=2:
            createDynamicData(j)#This call will create the dynamic data.
        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        tmpW.append(np.linalg.norm(x))
        allW.append(x)
        trainingMSE.append(calcTrErr(x, *fargs))
        validationMSE.append(calcValErr(x, *fargs))
        if j % np.ceil(nIterations/10) == 0:
            print('After iteration ',j,' training error(MSE/CE): ',trainingMSE[-1],' Validation error(MSE/CE): ',validationMSE[-1])
        j += 1
        # if iterationVariable is not None:
        #     iterationVariable.value = j

        ## If we get here, then we haven't terminated in the given number of
        ## iterations.

    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge",'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None}

def scgWithThresholding(x, f,gradf, calcTrErr, calcValErr,getIndices,*fargs, **params):
    """scg:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    tmpW = []
    trainingMSE = []
    validationMSE = []
    allW=[]

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.
    while j <= nIterations:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        tmpW.append(np.linalg.norm(x))
        allW.append(x)
        trainingMSE.append(calcTrErr(x, *fargs))
        validationMSE.append(calcValErr(x, *fargs))
        if j % np.ceil(nIterations/10) == 0:
            print('After iteration ',j,' training error(MSE/CE): ',trainingMSE[-1],' Validation error(MSE/CE): ',validationMSE[-1])
        if j > 250:
            splIndices=getIndices()
            tmpIndices=np.where(np.abs(x[splIndices])!=0) and np.where(np.abs(x[splIndices])<0.0000000001)[0]
            if len(tmpIndices)>0:
                print('No. of low values indices',len(tmpIndices),'. Setting those values to zeros')
                x[splIndices[tmpIndices]]=0
        j += 1
        # if iterationVariable is not None:
        #     iterationVariable.value = j

        ## If we get here, then we haven't terminated in the given number of
        ## iterations.

    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge",'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None}

def scgForBatch(x, f,gradf, calcTrMSE, calcValMSE,*fargs, **params):
    """scg:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.
    while j <= nIterations:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew

        if j % 25 == 0:
            print('After iteration ',j,' training MSE: ',trainingMSE[-1],' Validation MSE: ',validationMSE[-1])
        j += 1

    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge",'weightNorm':None,'trMSE':None,'valMSE':None,'weights':None}


def scgForDHCE(x, f, gradf, calcTrMSE, calcValMSE, calcClusteringError, *fargs, **params):#A function name 'calValErr' is being added to calculate MSE on validation data.
    '''
    Purpose:
    The following scg function is implemented with early stopping.
    Early stopping is done by evaluting the error on validation data.
    Validation error will be calculated for 20 consecutive iteration.
    I'm calling this as the error window. If the mean error of first
    1st half of the error window(1..10 itr) is less than the 2nd half
    (11..20) then the training will be stopped. This small modification
    is done by Tomojit Ghosh
    '''

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    trainingDone = False # Flag to inidicate whether training is complete or not. I'll use validation error to set the flag to 'True'
    validationMSE = []
    trainingMSE = []
    clusteringError=[]
    tmpW=[]
    windowSize=100
    #tmpW = deque(maxlen=windowSize)

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.

    #while j <= nIterations:
    while not(trainingDone):

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        #pdb.set_trace()
        validationMSE.append(calcValMSE(x, *fargs))
        trainingMSE.append(calcTrMSE(x, *fargs))
        clusteringError.append(calcClusteringError(x, *fargs))
        tmpW.append(x)
        if (j % 25) == 0:
            print('After iteration ',j,'MSE on training data:',trainingMSE[-1],' and validation data:',validationMSE[-1],'Clustering error:',clusteringError[-1])
        #Below is the stopping criteria
        if (j > windowSize) and (np.mean(clusteringError[-int(windowSize):-int(windowSize/2)]) <= np.mean(clusteringError[-int(windowSize/2):])):
            trainingDone = True
            #lowestValErrIndex = np.where(np.min(validationMSE[-int(windowSize):]) == validationMSE[-int(windowSize):])[0][0]
            pdb.set_trace()
            lowestClusteringErrIndex = np.where(np.min(clusteringError[-int(windowSize):]) == clusteringError[-int(windowSize):])[0][0]
            bestW = tmpW[lowestClusteringErrIndex]
            bestSCGItr = j-windowSize + lowestClusteringErrIndex +1
            #pdb.set_trace()
            break
        j += 1
        # if iterationVariable is not None:
        #     iterationVariable.value = j

        ## If we get here, then we haven't terminated in the given number of
        ## iterations.

    #return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
    #        'reason':"did not converge"}
    return {'x':bestW, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge",'trMSE':trainingMSE,'valMSE':validationMSE,'bestItr':bestSCGItr}


def scgWithEarlyStop(x, f, gradf, calcTrMSE, calcValMSE, windowSize, *fargs, **params):#A function name 'calValErr' is being added to calculate MSE on validation data.
    '''
    Purpose:
    The following scg function is implemented with early stopping.
    Early stopping is done by evaluting the error on validation data.
    Validation error will be calculated for 20 consecutive iteration.
    I'm calling this as the error window. If the mean error of first
    1st half of the error window(1..10 itr) is less than the 2nd half
    (11..20) then the training will be stopped. This small modification
    is done by Tomojit Ghosh
    '''

    from copy import copy
    from collections import deque

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    trainingDone = False # Flag to inidicate whether training is complete or not. I'll use validation error to set the flag to 'True'
    validationMSE = []
    trainingMSE = []
    tmpW = deque(maxlen=windowSize)

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.

    #while j <= nIterations:
    while not(trainingDone):

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        #pdb.set_trace()
        validationMSE.append(calcValMSE(x, *fargs))
        trainingMSE.append(calcTrMSE(x, *fargs))
        #validationMSE.append(calValErr())
        tmpW.append(x)
        if ((j+1) % 25) == 0:
            print('After iteration ',j,'MSE on training data:',trainingMSE[-1],' and validation data:',validationMSE[-1])
        #Below is the stopping criteria
        if j>= windowSize and (np.mean(validationMSE[-int(windowSize):-int(windowSize/2)]) <= np.mean(validationMSE[-int(windowSize/2):])):
            trainingDone = True
            lowestValErrIndex = np.where(np.min(validationMSE[-int(windowSize):]) == validationMSE[-int(windowSize):])[0][0]
            bestW = tmpW[lowestValErrIndex]
            bestSCGItr = j-windowSize + lowestValErrIndex +1
            #pdb.set_trace()
            break
        j += 1
        # if iterationVariable is not None:
        #     iterationVariable.value = j

        ## If we get here, then we haven't terminated in the given number of
        ## iterations.

    #return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
    #        'reason':"did not converge"}
    return {'x':bestW, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge",'trMSE':trainingMSE,'valMSE':validationMSE,'bestItr':bestSCGItr}


def scgWithErrorCutoff(x, f, gradf, calcTrErr, calcValErr,errThreshold=0.005, windowSize=5, *fargs, **params):
    '''
    Purpose:
    The following scg function is implemented with stopping criteria
    "error cutoff". This function will be used to stop scg automatically
    in pre-train step while training deep autoencoder. Error in the training
    data is captured for window which I'm calling error-window. And then
    the coefficient of variance(cv) of this window will be compared with a
    threshold value. Training will be stopped when cv <= threshold. A low
    value of cv means there is not enough error-decrease going on.

    This modification is done by Tomojit Ghosh.
    '''

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    trainingDone = False # Flag to inidicate whether training is complete or not. I'll use validation error to set the flag to 'True'
    trainingErr = []

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.

    #while j <= nIterations:
    while not(trainingDone):

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        trainingErr.append(calcTrErr(x, *fargs))
        if j % np.ceil(nIterations/10) == 0:
            print('After iteration ',j,'Training CE/MSE:',trainingErr[-1])
        #Below is the stopping criteria
        if trainingErr[-1] < errThreshold:
            trainingDone = True
            bestW = x
            break
        j += 1
    return {'x':bestW, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge", 'weightNorm':tmpW, 'trMSE':trainingErr}


def scgWithWeightCutoff(x, f, gradf, cvThreshold=0.005, windowSize=10, *fargs, **params):
    '''
    Purpose:
    The following scg function is implemented with stopping criteria
    "weight cutoff". This function will be used to stop scg automatically
    in pre-train step in training deep autoencoder. After each iteration
    the l2-norm will be calculated of the weight matrix. 10 consecutive
    norms will be calculated and then the coefficient of variance(cv) of this
    window will be compared with a threshold value. Training will be stopped
    when cv <= threshold. A low value of cv means there is not enough
    weight-update going on. So learning is being saturated.

    This modification is done by Tomojit Ghosh.
    '''

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    trainingDone = False # Flag to inidicate whether training is complete or not. I'll use validation error to set the flag to 'True'
    validationError = []
    tmpW = []

    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.

    #while j <= nIterations:
    while not(trainingDone):
        if j % 50 == 0:
            print('Iteration ',j,' completed...')

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        #validationError.append(calValErr(x, *fargs))
        tmpW.append(np.linalg.norm(x))
        #print('After iteration ',j,'Training MSE:',fnow,' Norm:',tmpW[-1])
        #Below is the stopping criteria

        if j>= windowSize and (np.std(tmpW[-windowSize:])/np.mean(tmpW[-windowSize:]) <= cvThreshold):
            #print('Current CV:',(np.std(tmpW[-windowSize:])/np.mean(tmpW[-windowSize:])))
            trainingDone = True
            bestW = x
            #pdb.set_trace()
            break
        j += 1
        # if iterationVariable is not None:
        #     iterationVariable.value = j

        ## If we get here, then we haven't terminated in the given number of
        ## iterations.

    #return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
    #        'reason':"did not converge"}
    return {'x':bestW, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge",'valErr':validationError, 'weightNorm':tmpW}


def dropConnect(x,dropPercentage):
    noOfDroppedConnect = int(np.ceil(len(x)*dropPercentage))
    dIndex = np.arange(len(x));
    np.random.shuffle(dIndex)
    dIndex = dIndex[:noOfDroppedConnect]
    dropConnectMat = np.ones([len(x)])
    dropConnectMat[dIndex] = 0
    return np.multiply(x,dropConnectMat),dropConnectMat

def antiMask(dropConnectMat):
    return (np.ones([len(dropConnectMat)]) - dropConnectMat)

def scgWithDropConnect(x, f,gradf, calcTrMSE, calcValMSE,dropPercentage,*fargs, **params):
    """scg:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    from copy import copy

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0)
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library

    #nvars = len(x)
    nvars = 2
    sigma0 = 1.0e-6
    #pdb.set_trace()
    hold_x = x
    x,dropConnectMat = dropConnect(x,dropPercentage) #dropping connections
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    tmpW = []
    trainingMSE = []
    validationMSE = []
    allW=[]


    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None

    ### Main optimization loop.
    while j <= nIterations:
        #pdb.set_trace()
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            #gplus = gradf(np.multiply(xplus,dropConnectMat), *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma
        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta

        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #fnew = f(np.multiply(xnew,dropConnectMat), *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        if verbose and (j % max(1,np.ceilnIterations/10)) == 0:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)

        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow

        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                #gradnew = gradf(np.multiply(x,dropConnectMat), *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        tmpW.append(np.linalg.norm(x))
        allW.append(x)
        trainingMSE.append(calcTrMSE(x, *fargs))
        validationMSE.append(calcValMSE(x, *fargs))
        if j % 1 == 0:
            print('Iteration ',j,' is complete. Training MSE:',trainingMSE[-1],'Validation MSE: ',validationMSE[-1])
        j += 1

    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge",'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW}

######################################################################
### steepest
def steepest(x, f,gradf, *fargs, **params):
    """steepest:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = steepest(firstx, parabola, parabolaGrad, center, S,
                 stepsize=0.01,xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    stepsize= params.pop("stepsize",0.1)
    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision", 1.e-8)  # 1.e-8 is a default value
    fPrecision = params.pop("fPrecision", 1.e-8)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)

    xtracep = True
    ftracep = True

    i = 1
    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None
    oldf = f(x,*fargs)
    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = f(x,*fargs)
    else:
        ftrace = None

    while i <= nIterations:
        g = gradf(x,*fargs)
        newx = x - stepsize * g
        newf = f(newx,*fargs)
        if (i % (nIterations/10)) == 0:
            print("Steepest: Iteration",i,"Error",evalFunc(newf))
        if xtracep:
            xtrace[i,:] = newx
        if ftracep:
            ftrace[i] = newf
        if np.any(newx == np.nan) or newf == np.nan:
            raise ValueError("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
        if np.any(newx==np.inf) or  newf==np.inf:
            raise ValueError("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
        if max(abs(newx - x)) < xPrecision:
            return {'x':newx, 'f':newf, 'nIterations':i, 'xtrace':xtrace[:i,:], 'ftrace':ftrace[:i],
                    'reason':"limit on x precision"}
        if abs(newf - oldf) < fPrecision:
            return {'x':newx, 'f':newf, 'nIterations':i, 'xtrace':xtrace[:i,:], 'ftrace':ftrace[:i],
                    'reason':"limit on f precision"}
        x = newx
        oldf = newf
        i += 1

    return {'x':newx, 'f':newf, 'nIterations':i, 'xtrace':xtrace[:i,:], 'ftrace':ftrace[:i], 'reason':"did not converge"}


if __name__ == "__main__":

    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])

    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)

    print('Stopped after',r['nIterations'],'iterations. Reason for stopping:',r['reason'])
    print('Optimal: point =',r['x'],'f =',r['f'])
