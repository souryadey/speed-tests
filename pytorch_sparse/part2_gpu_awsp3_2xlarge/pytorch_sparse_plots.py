import matplotlib.pyplot as plt
import numpy as np


def plot_sparse_diagonal(sizes, densecputimes, sparsecputimes, densegputimes, sparsegputimes, closeup_cutoff=6):
    results_folder = './'
    
    plt.figure(figsize = (9,9))
    plt.semilogx(sizes, densecputimes, basex=2, color='r', linewidth=2, label='Dense on CPU')
    plt.semilogx(sizes, sparsecputimes, basex=2, color='b', linewidth=2, label='Sparse on CPU')
    plt.semilogx(sizes, densegputimes, basex=2, color='k', linewidth=2, label='Dense on Cuda GPU')
    plt.semilogx(sizes, sparsegputimes, basex=2, color='g', linewidth=2, label='Sparse on Cuda GPU')
    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (sec)', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.legend(fontsize = 18)
    plt.savefig(results_folder+'sparse_diagonal', bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (5,5))
    plt.semilogx(sizes[:closeup_cutoff], densecputimes[:closeup_cutoff]*10**6, basex=2, color='r', linewidth=2, label='Dense on CPU')
    plt.semilogx(sizes[:closeup_cutoff], sparsecputimes[:closeup_cutoff]*10**6, basex=2, color='b', linewidth=2, label='Sparse on CPU')
    plt.semilogx(sizes[:closeup_cutoff], densegputimes[:closeup_cutoff]*10**6, basex=2, color='k', linewidth=2, label='Dense on Cuda GPU')
    plt.semilogx(sizes[:closeup_cutoff], sparsegputimes[:closeup_cutoff]*10**6, basex=2, color='g', linewidth=2, label='Sparse on Cuda GPU')
#    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (usec)', fontsize=16)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.gca().tick_params(axis='both',labelsize=13)
    plt.grid()
#    plt.legend(fontsize = 18)
    plt.savefig(results_folder+'sparse_diagonal_closeup', bbox_inches='tight')
    plt.show()
    
    
def plot_sparse_fixeddensity_12p5(sizes, densecputimes, sparsecputimes, densegputimes, sparsegputimes, closeup_cutoff=6):
    results_folder = './'
    
    plt.figure(figsize = (9,9))
    plt.semilogx(sizes, densecputimes, basex=2, color='r', linewidth=2, label='Dense on CPU')
    plt.semilogx(sizes, sparsecputimes, basex=2, color='b', linewidth=2, label='Sparse on CPU')
    plt.semilogx(sizes, densegputimes, basex=2, color='k', linewidth=2, label='Dense on Cuda GPU')
    plt.semilogx(sizes, sparsegputimes, basex=2, color='g', linewidth=2, label='Sparse on Cuda GPU')
    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (sec)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize = 18)
    plt.savefig(results_folder+'sparse_fixeddensity_12p5', bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (5,5))
    plt.semilogx(sizes[:closeup_cutoff], densecputimes[:closeup_cutoff]*10**6, basex=2, color='r', linewidth=2, label='Dense on CPU')
    plt.semilogx(sizes[:closeup_cutoff], sparsecputimes[:closeup_cutoff]*10**6, basex=2, color='b', linewidth=2, label='Sparse on CPU')
    plt.semilogx(sizes[:closeup_cutoff], densegputimes[:closeup_cutoff]*10**6, basex=2, color='k', linewidth=2, label='Dense on Cuda GPU')
    plt.semilogx(sizes[:closeup_cutoff], sparsegputimes[:closeup_cutoff]*10**6, basex=2, color='g', linewidth=2, label='Sparse on Cuda GPU')
#    plt.xlabel('Size (n) of nxn matrix', fontsize=16)
    plt.ylabel('Time (usec)', fontsize=16)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.gca().tick_params(axis='both',labelsize=13)
    plt.grid()
#    plt.legend(fontsize = 18)
    plt.savefig(results_folder+'sparse_fixeddensity_12p5_closeup', bbox_inches='tight')
    plt.show()
    

def plot_timevsdensity(times, size, title, axhline, multiplier=1):
    results_folder = './'
    nnzs = [2**i for i in range(2*int(np.log2(size))+1)]
    nnzs = nnzs[:len(times)] #adjust nnzs to have same size as times, such as for cases where all sims were not done
    plt.figure(figsize = (8,8))
    plt.semilogx(np.array(nnzs)*100/(size*size), times*multiplier, color='b', linewidth=2.5)
    plt.xlabel('Density (%) of non-zero elements', fontsize=16)
    plt.ylabel('Time ({0})'.format('ns' if multiplier==10**9 else 'usec' if multiplier==10**6 else 'ms' if multiplier=='10**3' else 'sec'), fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axvline(100/size, linestyle='--', linewidth=1.5, color='k') #diagonal matrix density
    plt.axvline(12.5, linestyle='--', linewidth=1.5, color='k') #12.5% density
    plt.axhline(axhline, linestyle='--', linewidth=2, color='r') #dense case time taken
    plt.grid()
    plt.title(title, fontsize=20)
    plt.savefig(results_folder+'timevsdensity_size{0}'.format(size), pad_inches=0)
    plt.show()
    
    


cpu = np.load('./cputimes.npz')
gpu = np.load('./gputimes.npz')

plot_sparse_diagonal(
                     sizes = [2**i for i in range(2,13)][1:],
                     densecputimes = cpu['dense_times'][1:],
                     sparsecputimes = cpu['sparse_diagonal_times'][1:],
                     densegputimes = gpu['densetimes'][1:],
                     sparsegputimes = gpu['sparse_diagonal_times'][1:]
                     )

plot_sparse_fixeddensity_12p5(
                              sizes = [2**i for i in range(2,13)][1:],
                              densecputimes = cpu['dense_times'][1:],
                              sparsecputimes = cpu['sparse_fixeddensity_12p5_times'][1:],
                              densegputimes = gpu['densetimes'][1:],
                              sparsegputimes = gpu['sparse_fixeddensity_12p5_times'][1:]
                              )

    
axhlines = [9.8361,11.7787, 148.3023,11.9619, 7.4373,0.17, 459.3107,9.5664] #cpu,gpu
for i,size,multiplier in zip(range(4), [64,256,1024,4096], [10**6,10**6,10**3,10**3]):
    plot_timevsdensity(cpu['sparse_fixedsize{0}_times'.format(size)], size, title = 'n={0} on CPU'.format(size), axhline=axhlines[2*i], multiplier=multiplier)
    plot_timevsdensity(gpu['sparse_fixedsize{0}_times'.format(size)], size, title = 'n={0} on CUDA GPU'.format(size), axhline=axhlines[2*i+1], multiplier=multiplier)
        