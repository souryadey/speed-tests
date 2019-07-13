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


#%% Plot results
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
plt.semilogx(sizes[:7], densetimes[:7]*10**6, basex=2, color='r', linewidth=2, label='Dense')
plt.semilogx(sizes[:7], sptimes[:7]*10**6, basex=2, color='b', linewidth=2, label='Sparse')
plt.xlabel('Size (n) of nxn matrix', fontsize=16)
plt.ylabel('Time (usec)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(fontsize = 18)
plt.savefig('bothvarying_closeup', bbox_inches='tight', pad_inches=0)
plt.show()



#%% Density fixed, size varying
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


#%% Plot results
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
plt.semilogx(sizes[:6], densetimes[:6]*10**6, basex=2, color='r', linewidth=2, label='Dense')
plt.semilogx(sizes[:6], sptimes[:6]*10**6, basex=2, color='b', linewidth=2, label='Sparse')
plt.xlabel('Size (n) of nxn matrix', fontsize=16)
plt.ylabel('Time (usec)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(fontsize = 18)
plt.savefig('fixeddensity_12p5_closeup', bbox_inches='tight', pad_inches=0)
plt.show()



#%% Size fixed, density varying
for size,axhline,multiplier,timescale in zip([16,128,1024,8192], [4.95,28,8.33,4.4], [10**6,10**6,10**3,1], ['usec','usec','ms','sec']):
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

