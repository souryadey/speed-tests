### Sourya Dey, USC
### Comparing sparse and dense matrices in Pytorch
### Pytorch uses Sparse COO format
### Multiplication time is used as metric
### Run on CUDA GPU if available, else CPU

import numpy as np
import timeit


# =============================================================================
# Dense with varying size x dense of same varying size
# =============================================================================
def dense(sizes = [2**i for i in range(2,13)]):
    times = np.zeros_like(sizes, dtype=float)
    
    for i,size in enumerate(sizes):
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        
        setup = (
                'import torch;'
                'size = {0};'
                'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
                'densemat = torch.randn(size,size).to(device);'
                'multmat = torch.randn(size,size).to(device)'
                ).format(size)
        reps, time = timeit.Timer(stmt = 'torch.mm(densemat,multmat)', setup = setup).autorange()
        times[i] = time/reps
    
    return times



# =============================================================================
# Diagonal sparse with varying size x dense of same varying size
# =============================================================================
def sparse_diagonal(sizes = [2**i for i in range(2,13)]):
    times = np.zeros_like(sizes, dtype=float)
    
    for i,size in enumerate(sizes):
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        
        setup = (
                'import torch;'
                'size = {0};'
                'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
                'spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.arange(size).tolist(),torch.arange(size).tolist()]), values = torch.randn(size), size=[size,size], dtype=torch.float32, device=device);'
                'multmat = torch.randn(size,size).to(device)'
                ).format(size)
        reps, time = timeit.Timer(stmt = 'torch.sparse.mm(spmat,multmat)', setup = setup).autorange()
        times[i] = time/reps
    
    return times



# =============================================================================
# Sparse with density fixed at 12.5% and varying size x dense of same varying size
# =============================================================================
def sparse_fixeddensity_12p5(sizes = [2**i for i in range(2,13)]):
    times = np.zeros_like(sizes, dtype=float)
    
    for i,size in enumerate(sizes):
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        
        setup = (
                'import torch;'
                'size = {0};'
                'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
                'nnz = size*size//8;'
                'spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.randint(size,(nnz,)).tolist(),torch.randint(size,(nnz,)).tolist()]), values = torch.randn(nnz), size=[size,size], dtype=torch.float32, device=device);'
                'multmat = torch.randn(size,size).to(device)'
                ).format(size)
        reps, time = timeit.Timer(stmt = 'torch.sparse.mm(spmat,multmat)', setup = setup).autorange()
        times[i] = time/reps
        
    return times



# =============================================================================
# Sparse with certain fixed sizes and density varying x dense of same fixed sizes
# =============================================================================
def sparse_fixedsizes(size):
    print('SIZE = {0}'.format(size))
    nnzs = [2**i for i in range(2*int(np.log2(size))+1)]
    if size==4096:
        nnzs = nnzs[:-2] #prevent excessive time being taken
    times = np.zeros_like(nnzs, dtype=float)
    
    for i,nnz in enumerate(nnzs):
        print('Density = {0}%'.format(nnz*100/(size*size)))
        
        setup = (
            'import torch;'
            'size = {0};'
            'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
            'nnz = {1};'
            'spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.randint(size,(nnz,)).tolist(),torch.randint(size,(nnz,)).tolist()]), values = torch.randn(nnz), size=[size,size], dtype=torch.float32, device=device);'
            'multmat = torch.randn(size,size).to(device)'
            ).format(size,nnz)
        reps, time = timeit.Timer(stmt = 'torch.sparse.mm(spmat,multmat)', setup = setup).autorange()
        times[i] = time/reps

    return times
        
        

# =============================================================================
# Main execution
# =============================================================================
dense_times = dense()
sparse_diagonal_times = sparse_diagonal()
sparse_fixeddensity_12p5_times = sparse_fixeddensity_12p5()
sparse_fixedsize64_times = sparse_fixedsizes(64)
sparse_fixedsize256_times = sparse_fixedsizes(256)
sparse_fixedsize1024_times = sparse_fixedsizes(1024)
sparse_fixedsize4096_times = sparse_fixedsizes(4096)
np.savez_compressed('times.npz',
                    dense_times=dense_times,
                    sparse_diagonal_times=sparse_diagonal_times,
                    sparse_fixeddensity_12p5_times=sparse_fixeddensity_12p5_times,
                    sparse_fixedsize64_times = sparse_fixedsize64_times,
                    sparse_fixedsize256_times = sparse_fixedsize256_times,
                    sparse_fixedsize1024_times = sparse_fixedsize1024_times,
                    sparse_fixedsize4096_times = sparse_fixedsize4096_times
                    )


