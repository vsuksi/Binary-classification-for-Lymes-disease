def limma(data, batch_labels):
    '''
    A pure numpy implementation of the code found at:

    https://github.com/chichaumiau/removeBatcheffect/blob/master/limma.py

    based on our weak understanding of what the patsy package does and
    following along with the example in the link above. Their data comes from

    https://github.com/brentp/combat.py

    Note that we don't have the capability to include an assumed model effect
    in the covariance_matrix as in Chichau's version, but our approach is
    only to remove batch factors, then apply machine learning algo's to the
    result. Making an initial assumption of a linear model in the phenotypes in
    preprocessing stage may not be appropriate depending on the machine learning
    tools used later in the pipeline.

    Inputs:
        data: m-by-n array of data; m observations in R^n.
        batch_labels: m array of labels. Assumed to be discrete; string or
            other labels are handled cleanly.
    Outputs:
        data: Modified data matrix.

    There are options in the original limma.removeBatchEffect() code
    and corresponding limma_chichau() function (see in calcom/utils/limma.py)
    which aren't implemented in this version.
    '''
    import numpy as np

    m,n = data.shape

    unique_batches = np.unique(batch_labels)
    bmap = {b:i for i,b in enumerate(unique_batches)}
    nbatches = len(np.unique(batch_labels))

    design_matrix = np.zeros((m,nbatches))
    for i,b in enumerate(batch_labels):
        design_matrix[i,bmap[b]] = 1
    #

    # Idea here is that the states of each of the labels is encoded
    # in R^{nbatches-1} where each cardinal direction represents a
    # batch, and the origin is the first batch.
    design_matrix = design_matrix[:,1:]

    # Seems to insert a row of -1's for any sample in the first batch.
    rowsum = design_matrix.sum(axis=1) -1
    design_matrix=(design_matrix.T+rowsum).T

    # Apparently this is the "null" model generated; just a ones vector.
    covariate_matrix = np.ones((m,1))
    design_batch = np.hstack( (covariate_matrix,design_matrix) )

    # coefficients, _, _, _ = np.linalg.lstsq(design_batch,data)
    # Need to silence an annoying warning.
    numpy_version = '.'.join( np.__version__.split('.')[:2] )
    if numpy_version < '1.14':
        coefficients, _, _, _ = np.linalg.lstsq(design_batch,data)
    else:
        coefficients, _, _, _ = np.linalg.lstsq(design_batch,data,rcond=None)
    #

    # Subtract off the component of this least squares linear model
    # whose contribution is due to batch effect.
    return data - np.dot(design_matrix,coefficients[-(nbatches-1):])
#

def get_linear_batch_shifts(data,batch_labels, tolerate_nans=False):
    '''
    Given a data matrix and associated batch_labels, return 
    a dictionary mapping the list of batch labels to the
    associated shifts used in linear batch normalization (limma).

    Inputs:
        data : numpy array shape (n,d)
        labels : array shape (n,); integers or strings, of batch labels.
    Outputs:
        shift_dict : dictionary whose keys are a unique list of 
            batches, and values are class-specific shifts for each batch.
    '''
    import numpy as np

    if tolerate_nans:
        op = np.nanmean
    else:
        op = np.mean
    #

    blist = np.unique(batch_labels)
    ec = {batch: np.where(batch_labels==batch)[0] for batch in blist}

    shift_dict = {batch: op(np.array(data)[ec[batch]], axis=0) for batch in blist}
    net_mean = op( list(shift_dict.values()), axis=0)

    for b in blist:
        shift_dict[b] -= net_mean
    #
    
    return shift_dict
#

def limma_R(probes, labels):
    """
    Takes probe and label arrays and performs limma normalization for batch
    correction and return the normalized data. This function should be called inside
    a try catch block which handles an EnvironmentException.

    This is an interface to limma.removeBatchEffect() using a
    python wrapper for R (rpy2).

    Args:
        -probes: combined dataset
        -labels: study labels; not classification labels

    Returns:
        -normalized data if R installed; throws exception otherwise

    """
    # Make sure that R is installed
    # Otherwise return the original data
    from subprocess import Popen, PIPE
    proc = Popen(["which", "R"],stdout=PIPE,stderr=PIPE)
    exit_code = proc.wait()
    if exit_code != 0:
        raise EnvironmentError("R not installed.")
    #
    from rpy2 import robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.numpy2ri import numpy2ri #import the module for converting numpy arrays to R

    study_labels = labels

    all_data = probes.T

    placeholder = ro.conversion.py2ri #save the default converter for later
    ro.conversion.py2ri = numpy2ri
    r_data=ro.Matrix(all_data) #load in the data and the strings determining studies
    r_studies=ro.Vector(study_labels)
    ro.conversion.py2ri = placeholder #reset converter
    r_limma = ro.packages.importr('limma') #import limma

    new_data = np.array(r_limma.removeBatchEffect(r_data, r_studies))
    return new_data.T

'''
def limma(pheno, exprs, covariate_formula, design_formula='1', rcond=1e-8):
    design_matrix = patsy.dmatrix(design_formula, pheno)

    design_matrix = design_matrix[:,1:]
    rowsum = design_matrix.sum(axis=1) -1
    design_matrix=(design_matrix.T+rowsum).T

    covariate_matrix = patsy.dmatrix(covariate_formula, pheno)
    design_batch = np.hstack((covariate_matrix,design_matrix))
    coefficients, res, rank, s = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
    beta = coefficients[-design_matrix.shape[1]:]
    return exprs - design_matrix.dot(beta).T
'''

def limma_chichau(pheno, exprs, covariate_formula, design_formula='1', rcond=1e-8):
    '''
    A pure python implementation of limma's removeBatchEffect found online at:

    https://github.com/chichaumiau/removeBatcheffect/blob/master/limma.py

    Requires the patsy library, and assumes the experimental data is organized
    in terms of dictionaries, where entries are a set of observations
    in terms of each variable (e.g., exprs['probe1'], exprs['probe2'], etc)
    and the collection of all label (phenotype) information is in "pheno".
    '''
    import patsy
    import numpy as np

    design_matrix = patsy.dmatrix(design_formula, pheno)

    design_matrix = design_matrix[:,1:]
    rowsum = design_matrix.sum(axis=1) -1
    design_matrix=(design_matrix.T+rowsum).T

    covariate_matrix = patsy.dmatrix(covariate_formula, pheno)
    design_batch = np.hstack((covariate_matrix,design_matrix))
    coefficients, res, rank, s = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
    beta = coefficients[-design_matrix.shape[1]:]
    return exprs - design_matrix.dot(beta).T
#

if __name__ == "__main__":

    # probes = np.array([[10,21,32],[12,21,30],[11,22,31],[11,21,32],[20,41,62],[22,41,60],[21,40,62],[30,61,89]])
    # batch_labels = np.array([[0,0,0,0,2,2,2,1]])
    # bl = np.array([[1,1,1,1,1,1,1,1],[1,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,1,1,1,0]])

    # new_data = limma(probes, batch_labels)
    # print(new_data)
    #
    # # does not match the results
    # new_data = limma1(probes, batch_labels)
    # print(new_data)

    #######
    import pandas as pd
    import numpy as np

    pheno=pd.read_table('data/bladder-pheno.txt', index_col=0)
    exprs=pd.read_table('data/bladder-expr.txt', index_col=0)

    data = np.array(exprs).T # Row-major; each row is an observation across all genes
    batch_labels = np.array(pheno['batch'])

    # Lange's R interface
    rs = limma_R(data,batch_labels)
    rs = rs

    # Chichau's Python implementation using patsy and numpy
    chichau = limma_chichau(pheno, exprs, "", "C(batch)")
    chichau = np.array(chichau)

    # Our Python implementation using numpy alone
    ours = limma(data,batch_labels)

    results = [rs,chichau.T,ours]   # Column vs row major

    diffmat = np.array([[ np.linalg.norm(results[i]-results[j])/np.linalg.norm(exprs) for i in range(3)] for j in range(3)])

    print("pairwise relative norms between resulting matrices between implementations:")
    print(diffmat)

#
