'''
Sourya Dey, USC

Compare multiplication time of sparse x sparse (both densities varying individually)
using only regular dense routines in Pytorch, i.e. special sparse routines are NOT used
'''

import timeit
import numpy as np
import matplotlib.pyplot as plt


def matmult_random(a=1000,b=1000,c=1000, density1=1, density2=1):
    '''
    Mat 1: axb, Mat 2: bxc, Result: axc
    Each has non-zero elements = random numbers
    '''
    setup = (
            'import numpy as np;'
            'import torch;'
            'from adjmatint import adjmat_random;'
            'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
            'mat1 = torch.randn({0},{1}, dtype=torch.float32, device=device) * torch.as_tensor(adjmat_random(density={3},p={1},n={0}), dtype=torch.float32, device=device);'
            'mat2 = torch.randn({1},{2}, dtype=torch.float32, device=device) * torch.as_tensor(adjmat_random(density={4},p={2},n={1}), dtype=torch.float32, device=device);'
            ).format(a,b,c,density1,density2)
    reps, time = timeit.Timer(stmt = 'torch.matmul(mat1,mat2)', setup=setup).autorange()
    return time/reps


def matvecmult_random(a=1000,b=1000, density=1):
    '''
    Mat: axb, Vec: bx1, Result: ax1
    Each has non-zero elements = random numbers
    '''
    setup = (
            'import numpy as np;'
            'import torch;'
            'from adjmatint import adjmat_random;'
            'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
            'mat = torch.randn({0},{1}, dtype=torch.float32, device=device) * torch.as_tensor(adjmat_random(density={2},p={1},n={0}), dtype=torch.float32, device=device);'
            'vec = torch.randn({1},1, dtype=torch.float32, device=device);'
            ).format(a,b,density)
    reps, time = timeit.Timer(stmt = 'torch.matmul(mat,vec)', setup=setup).autorange()
    return time/reps


def matmult_clashfree(a=1000,b=1000,c=1000, density1=1, density2=1):
    '''
    Mat 1: axb, Mat 2: bxc, Result: axc
    Each has non-zero elements = random numbers
    '''
    setup = (
            'import numpy as np;'
            'import torch;'
            'from adjmatint import adjmat_clash_free;'
            'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");'
            'mat1 = torch.randn({0},{1}, dtype=torch.float32, device=device) * torch.as_tensor(adjmat_clash_free(p={1}, fo=int(np.ceil({3}*{0})), n={0}, z={1}//10), dtype=torch.float32, device=device);'
            'mat2 = torch.randn({1},{2}, dtype=torch.float32, device=device) * torch.as_tensor(adjmat_clash_free(p={2}, fo=int(np.ceil({4}*{1})), n={1}, z={2}//10), dtype=torch.float32, device=device)'
            ).format(a,b,c,density1,density2)
    reps, time = timeit.Timer(stmt = 'torch.matmul(mat1,mat2)', setup=setup).autorange()
    return time/reps


def plot_results(times, filename, multiplier=10**6,
                 density1s = [1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1.],
                 density2s = [1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1.]):
    colors = ['k','b','r','g','brown','purple','magenta']
    plt.figure(figsize=(10,10))
    for i,density1 in enumerate(density1s):
        plt.semilogx(density2s,times[i]*multiplier, label=density1, color=colors[i])
    plt.xlabel('Mat2 density', fontsize=16)
    plt.ylabel('Time ({0})'.format('sec' if multiplier==1 else 'ms' if multiplier==10**3 else 'us'), fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(which='both')
    leg = plt.legend(fontsize=16, ncol=2)
    leg.set_title('Mat1 density', prop={'size':16})
    plt.savefig('./{0}'.format(filename), bbox_inches='tight')
    plt.show()



# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    a,b,c = 1000,1000,1000
    
    # regular order
    density1s = [1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1.]
    density2s = [1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1.]
    
    # reversed order
    density1s = [1.,5e-1,2e-1,1e-1,5e-2,2e-2,1e-2]
    density2s = [1.,5e-1,2e-1,1e-1,5e-2,2e-2,1e-2]
    
    times = np.zeros((len(density1s),len(density2s)))
    
    for i,density1 in enumerate(density1s):
        for j,density2 in enumerate(density2s):
            times[i,j] = matmult_clashfree(a,b,c, density1,density2)
            print('{0}: {1}'.format(density2, times[i,j]))
        print()
    
    np.savez_compressed('clashfree_1Kx1K_gpu_reversedorder.npz', times=times, density1s=density1s, density2s=density2s)
    
    plot_results(times, filename='clashfree_1Kx1K_gpu_reversedorder', multiplier=10**6, density1s=density1s, density2s=density2s)
