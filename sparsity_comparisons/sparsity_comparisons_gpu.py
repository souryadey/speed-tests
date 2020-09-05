### Sourya Dey, USC
### Comparing different sparsity methods
### Uses Pytorch
### Multiplication time is used as metric
### Run on CUDA GPU if available, else CPU

import timeit
import pickle


sizes = [64,256,1024]
times = {}

for size in sizes:
    nnzrows = [size//64,size//16,size//4]   
    for nnzrow in nnzrows:
        print('size{0}x{0}_nnzrow{1}.npy'.format(size,nnzrow))
            
        ## Case 1: Structured sparsity
        setup = (
                'import numpy as np;'
                'import torch;'
                'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
                'nnz = int({0}*{1});'
                'elems = np.random.randn(nnz);'
                'densemat = torch.randn({0},{0}).to(device);'
                'f = open("./readymade_adjmats/adjmat_size{0}x{0}_nnzrow{1}.npy", "rb");'
                'adj = np.load(f);'
                'f.close();'
                'adjf = adj.flatten();'
                'adjf[np.where(adjf==1)[0]] = elems;'
                'spmat = torch.from_numpy(adjf.reshape({0},{0})).float().to(device)'
                ).format(size,nnzrow)
        reps, time = timeit.Timer(stmt = 'torch.mm(spmat,densemat)', setup = setup).autorange()
        times['size{0}x{0}_nnzrow{1}'.format(size,nnzrow)] = [time/reps*(10**6)]
        
        ## Case 2: Random sparsity
        setup = (
                'import numpy as np;'
                'import torch;'
                'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
                'nnz = int({0}*{1});'
                'elems = np.random.randn(nnz);'
                'densemat = torch.randn({0},{0}).to(device);'
                'locs = np.random.choice({0}*{0},nnz, replace=False);'
                'spmat = np.zeros({0}*{0});'
                'spmat[locs] = elems;'
                'spmat = torch.from_numpy(spmat.reshape({0},{0})).float().to(device)'
                ).format(size,nnzrow)
        reps, time = timeit.Timer(stmt = 'torch.mm(spmat,densemat)', setup = setup).autorange()
        times['size{0}x{0}_nnzrow{1}'.format(size,nnzrow)].append(time/reps*(10**6))
        
        ## Case 3: Maximally unstructured sparsity
        setup = (
                'import numpy as np;'
                'import torch;'
                'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
                'nnz = int({0}*{1});'
                'elems = np.random.randn(nnz);'
                'densemat = torch.randn({0},{0}).to(device);'
                'spmat = np.zeros({0}*{0});'
                'spmat[:nnz] = elems;'
                'spmat = torch.from_numpy(spmat.reshape({0},{0})).float().to(device)'
                ).format(size,nnzrow)
        reps, time = timeit.Timer(stmt = 'torch.mm(spmat,densemat)', setup = setup).autorange()
        times['size{0}x{0}_nnzrow{1}'.format(size,nnzrow)].append(time/reps*(10**6))
        
        

with open('./times.pkl','wb') as f:
    pickle.dump(times,f)


