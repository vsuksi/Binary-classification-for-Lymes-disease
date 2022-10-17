from __future__ import absolute_import, division, print_function

try:
    import torch
except ImportError:
    torch = None
from numpy import finfo
    
def pytorch_matrix_rank(M, **kwargs):
    """    
    Return matrix rank of array using SVD method
    Rank of the array is the number of singular values of the array that are
    greater than `tol`.
    
    Implemented in accordance with - 
        https://github.com/numpy/numpy/blob/v1.14.2/numpy/linalg/linalg.py#L1608
    
    Parameters
    ----------
    M : array_like
        input vector or stack of matrices
    use_cuda : boolean (Default: False), optional
        if cuda is enabled, it is expected that the input matrix is already in the GPU
    """
    use_cuda = kwargs.get('use_cuda',False)

    # raise exception if GPU computation is requested but not avaiable
    if not (torch and torch.cuda.is_available()) and use_cuda:
        raise Exception("torch unavailable")
    
    # if cuda enabled, it is expect that the Matrix is already in the GPU
    if M.dim() < 2:
        return int( not (M == 0).sum() == M.nelement() )

    # display a rough warning if the torch version looks old
    if torch.__version__ <="1.2.0":
        print('torch version %s looks old; svd may fail. Continuing anyway.'%torch.__version__)
    #

    _, S, _ = torch.svd(M)

    tol = S.max() * max(M.size()[0],M.size()[1]) * finfo(float).eps

    return (S > tol).nonzero().size()[0]


if __name__ == "__main__":
    import numpy as np
    A = np.array([[1,2,3],[5,2,6],[2,3,1],[3,3,3]])
    A_c = torch.from_numpy(A).float().cuda()
    print(np.linalg.matrix_rank(A))
    print(pytorch_matrix_rank(A_c))



    
