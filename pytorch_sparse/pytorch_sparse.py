### Sourya Dey, USC
### Comparing sparse and dense matrices in Pytorch
### Pytorch uses Sparse COO format
### Multiplication time is used as metric


import torch
import numpy as np
import matplotlib.pyplot as plt
import timeit


###### Set device ######
device = 'cpu'


#%% Both size and density varying
def bothvarying():
    sizes = [2**i for i in range(1,14)]
    densetimes = np.zeros_like(sizes, dtype=float)
    sptimes = np.zeros_like(sizes, dtype=float)
    for i,size in enumerate(sizes):
        rep = 1000 if size<1000 else 100 if size<8000 else 10
        spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.arange(size).tolist(),torch.arange(size).tolist()]), values = torch.randn(size), size=[size,size], dtype=torch.float32, device=device)
        densemat = spmat.to_dense()
        multmat = torch.randn(size,size).to(device)
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        densetimes[i] = timeit.timeit('torch.mm(densemat,multmat)', number=rep, globals=globals())/rep
        sptimes[i] = timeit.timeit('torch.sparse.mm(spmat,multmat)', number=rep, globals=globals())/rep
        print()
        return sizes,densetimes,sptimes


#%% Plot results
def plot_bothvarying(densetimes, sptimes, sizes = [2**i for i in range(1,14)], closeup_cutoff=7):
    plt.figure(figsize = (8,8))
    plt.semilogx(sizes, densetimes, basex=2, color='r', linewidth=2, label='Dense')
    plt.semilogx(sizes, sptimes, basex=2, color='b', linewidth=2, label='Sparse')
    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (sec)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize = 18)
    plt.savefig('bothvarying', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    plt.figure(figsize = (8,8))
    plt.semilogx(sizes[:closeup_cutoff], densetimes[:closeup_cutoff]*10**6, basex=2, color='r', linewidth=2, label='Dense')
    plt.semilogx(sizes[:closeup_cutoff], sptimes[:closeup_cutoff]*10**6, basex=2, color='b', linewidth=2, label='Sparse')
    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (usec)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize = 18)
    plt.savefig('bothvarying_closeup', bbox_inches='tight', pad_inches=0)
    plt.show()



#%% Density fixed, size varying
def fixeddensity_12p5():
    sizes = [2**i for i in range(1,14)]
    densetimes = np.zeros_like(sizes, dtype=float)
    sptimes = np.zeros_like(sizes, dtype=float)
    for i,size in enumerate(sizes):
        rep = 1000 if size<1000 else 100 if size<2000 else 10 if size<8000 else 4
        nnz = size*size//8
        spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.randint(size,(nnz,)).tolist(),torch.randint(size,(nnz,)).tolist()]), values = torch.randn(nnz), size=[size,size], dtype=torch.float32, device=device)
        densemat = spmat.to_dense()
        multmat = torch.randn(size,size).to(device)
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        densetimes[i] = timeit.timeit('torch.mm(densemat,multmat)', number=rep, globals=globals())/rep
        sptimes[i] = timeit.timeit('torch.sparse.mm(spmat,multmat)', number=rep, globals=globals())/rep
        print()
        return sizes,densetimes,sptimes


#%% Plot results
def plot_fixeddensity_12p5(densetimes, sptimes, sizes = [2**i for i in range(1,14)], closeup_cutoff=6):
    plt.figure(figsize = (8,8))
    plt.semilogx(sizes, densetimes, basex=2, color='r', linewidth=2, label='Dense')
    plt.semilogx(sizes, sptimes, basex=2, color='b', linewidth=2, label='Sparse')
    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (sec)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize = 18)
    plt.savefig('fixeddensity_12p5', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    plt.figure(figsize = (8,8))
    plt.semilogx(sizes[:closeup_cutoff], densetimes[:closeup_cutoff]*10**6, basex=2, color='r', linewidth=2, label='Dense')
    plt.semilogx(sizes[:closeup_cutoff], sptimes[:closeup_cutoff]*10**6, basex=2, color='b', linewidth=2, label='Sparse')
    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (usec)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize = 18)
    plt.savefig('fixeddensity_12p5_closeup', bbox_inches='tight', pad_inches=0)
    plt.show()



#%% Size fixed, density varying
def timevsdensity(sizes=[16,128,1024,8192]):
    alltimes = []
    for size in sizes:
        print('Size = {0}'.format(size))
        nnzs = [2**i for i in range(2*int(np.log2(size))+1)]
        times = np.zeros_like(nnzs, dtype=float)
        multmat = torch.randn(size,size).to(device)
        for i,nnz in enumerate(nnzs):
            rep = 1000 if size in [16,128] else 100 if size==1024 else 10 if size==8192 and nnz<2**24 else 3
            spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.randint(size,(nnz,)).tolist(),torch.randint(size,(nnz,)).tolist()]), values = torch.randn(nnz), size=[size,size], dtype=torch.float32, device=device)
            print('Density = {0}%'.format(nnz*100/(size*size)))
            times[i] = timeit.timeit('torch.sparse.mm(spmat,multmat)', number=rep, globals=globals())/rep
            print()
        alltimes.append(times)
    return alltimes


#%% Plot results
def plot_timevsdensity(alltimes, sizes=[16,128,1024,8192], axhlines=[4.95,28,8.33,4.4], multipliers=[10**6,10**6,10**3,1], timescales=['usec','usec','ms','sec']):
    for times,size,axhline,multiplier,timescale in zip(alltimes,sizes,axhlines,multipliers,timescales):
        nnzs = [2**i for i in range(2*int(np.log2(size))+1)]
        plt.figure(figsize = (8,8))
        plt.semilogx(np.array(nnzs)*100/(size*size), times*multiplier, color='b', linewidth=2.5)
        plt.xlabel('Density (%) of non-zero elements', fontsize=16)
        plt.ylabel('Time ({0})'.format(timescale), fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.axvline(100/size, linestyle='--', linewidth=1.5, color='k')
        plt.axvline(12.5, linestyle='--', linewidth=1.5, color='k')
        plt.axhline(axhline, linestyle='--', linewidth=2, color='r')
        plt.grid()
        plt.savefig('timevsdensity_size{0}'.format(size), pad_inches=0)
        plt.show()
        
        
        
#%% RESULTS:

## For CPU results, see notebook pytorch_sparse.ipynb
        
## CUDA results:
plot_bothvarying(densetimes = np.array([1.20114120e-05, 1.17211800e-05, 1.21673840e-05,
 1.24037170e-05, 1.22183720e-05, 1.21485870e-05, 1.20420580e-05,
 1.18975290e-05, 1.94571400e-05, 1.91976200e-05, 2.18325600e-05,
 8.98779002e-05]), sptimes = np.array([4.45773923e-04, 4.47661231e-04, 4.51187064e-04,
 4.57516748e-04, 4.64012311e-04, 4.79302653e-04, 5.24293609e-04,
 7.58509805e-04, 1.81601162e-03, 8.84551543e-03, 5.86113757e-02,
 4.53782019e-01]), sizes = [2**i for i in range(2,14)], closeup_cutoff=9)
    
plot_fixeddensity_12p5(densetimes = np.array([1.16820430e-05, 1.19576200e-05, 1.21040890e-05,
 1.21376260e-05, 1.23550570e-05, 1.21628980e-05, 1.22118970e-05,
 1.23291610e-05, 1.95394600e-05, 2.48739001e-05, 2.67300999e-05,
 4.12047502e-05]), sptimes = np.array([4.20073598e-04, 4.20169182e-04, 4.25311569e-04,
 4.42686692e-04, 5.09491952e-04, 6.51234861e-04, 1.01269189e-03,
 2.66764489e-03, 1.29870602e-02, 6.05526627e-02, 3.35274848e-01,
 2.22102934e+00]), sizes = [2**i for i in range(2,14)], closeup_cutoff=6)
    
plot_timevsdensity(alltimes = [np.array([0.00012688, 0.00042909, 0.00042905, 0.00042981, 0.00043214, 0.00043417,
 0.00043727, 0.00044172, 0.00044566]), np.array([0.00012764, 0.00043028, 0.00042784, 0.00042907, 0.00043241, 0.00043647,
 0.00043842, 0.00044239, 0.00045139, 0.00047068, 0.00050599, 0.00059378,
 0.00070474, 0.00085024, 0.00107322]), np.array([0.00013225, 0.00076444, 0.00077957, 0.0007455, 0.00073285, 0.00073008,
 0.00071979, 0.00073976, 0.00074868, 0.00078344, 0.00082211, 0.00092456,
 0.00107505, 0.00135493, 0.00189982, 0.00302716, 0.00536672, 0.01216939,
 0.0226268, 0.04084562, 0.0660029]), np.array([1.44140500e-04, 3.58677376e-02, 3.58651663e-02, 3.59203537e-02,
 3.58924999e-02, 3.57446884e-02, 3.57195892e-02, 3.58683372e-02,
 3.59502009e-02, 3.63276156e-02, 3.69385350e-02, 3.77011554e-02,
 3.88156542e-02, 3.99834076e-02, 4.13225999e-02, 4.40332828e-02,
 4.86232494e-02, 6.24523949e-02, 9.17562767e-02, 1.47210463e-01,
 2.61094865e-01, 5.01124331e-01, 9.69065327e-01, 1.77367785e+00,
 3.26617806e+00, 5.59743446e+00, 8.99486859e+00])], sizes=[16,128,1024,8192], axhlines=[12.13,12.15,0.0195,6.5e-5], multipliers=[10**6,10**6,10**3,1], timescales=['usec','usec','ms','sec'])
        
        
        
        
        

