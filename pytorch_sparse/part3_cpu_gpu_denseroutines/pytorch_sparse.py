'''
Sourya Dey, USC

Compare multiplication time of sparse x sparse (both densities varying individually)
using only regular dense routines in Pytorch, i.e. special sparse routines are NOT used

Case 1: a=100 x b=100 times b=100 x c=100
Case 2: 8192x1024 x 1024x8192

Run on CUDA GPU if available, else CPU
'''

import timeit
import numpy as np
import matplotlib.pyplot as plt


def matmult_random(a=100,b=100,c=100, density1=1, density2=1):
    setup = (
            'import numpy as np;'
            'import torch;'
            'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
            'flattened_size = {0}*{1};'
            'mat1 = torch.randn(flattened_size);'
            'locs = np.random.choice(flattened_size, size=int(np.ceil((1-{3})*flattened_size)), replace=False);'
            'mat1[locs] = 0.;'
            'mat1 = mat1.view({0},{1}).to(device);'
            'flattened_size = {1}*{2};'
            'mat2 = torch.randn(flattened_size);'
            'locs = np.random.choice(flattened_size, size=int(np.ceil((1-{4})*flattened_size)), replace=False);'
            'mat2[locs] = 0.;'
            'mat2 = mat2.view({1},{2}).to(device)'
            ).format(a,b,c,density1,density2)
    reps, time = timeit.Timer(stmt = 'torch.mm(mat1,mat2)', setup=setup).autorange()
    return time/reps


def matmult_clashfree(a=100,b=100,c=100, density1=1, density2=1):
    setup = (
            'import numpy as np;'
            'import torch;'
            'from adjmatint import adjmat_clash_free;'
            'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
            'mat1 = torch.randn({0},{1}) * torch.as_tensor(adjmat_clash_free(p={1}, fo=int(np.ceil({3}*{0})), n={0}, z={1}//2), dtype=torch.float32, device=device);'
            'mat2 = torch.randn({1},{2}) * torch.as_tensor(adjmat_clash_free(p={2}, fo=int(np.ceil({4}*{1})), n={1}, z={2}//2), dtype=torch.float32, device=device)'
            ).format(a,b,c,density1,density2)
    reps, time = timeit.Timer(stmt = 'torch.matmul(mat1,mat2)', setup=setup).autorange()
    return time/reps


def plot_results(times, filename, multiplier=10**6, density1s=[0.01,0.02,0.05,0.1,0.2,0.5,1], density2s=[0.01,0.02,0.05,0.1,0.2,0.5,1]):
    colors = ['k','b','r','g','brown','purple','magenta']
    plt.figure(figsize=(10,10))
    for i,density1 in enumerate(density1s):
        plt.plot(density2s,times[i]*multiplier, label=density1, color=colors[i])
    plt.xlabel('Mat2 density', fontsize=16)
    plt.ylabel('Time ({0})'.format('sec' if multiplier==1 else 'ms' if multiplier==10**3 else 'us'), fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    leg = plt.legend(fontsize=16, ncol=2)
    leg.set_title('Mat1 density', prop={'size':16})
    plt.savefig('./{0}'.format(filename), bbox_inches='tight')
    plt.show()



# =============================================================================
# Main execution
# =============================================================================
a,b,c = 100,100,100
density1s = [0.01,0.02,0.05,0.1,0.2,0.5,1]
density2s = [0.01,0.02,0.05,0.1,0.2,0.5,1]
times = np.zeros((len(density1s),len(density2s)))

for i,density1 in enumerate(density1s):
    for j,density2 in enumerate(density2s):
        print(density2)
        times[i,j] = matmult_random(a,b,c, density1,density2)

np.savez_compressed('random_100x100_cputimes.npz', times=times)

#times = np.load('./random_8K1Kx1K8K_gputimes.npz')['times']
plot_results(times, filename='random_100x100_cpu', multiplier=10**6, density1s=density1s, density2s=density2s)
