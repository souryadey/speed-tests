# speed-tests

This is a repository to test and compare speed and efficiency of different programming methods and frameworks.

computation_frameworks -- Compare numpy, torch and tensorflow and Python's native lists

pytorch_sparse -- Compare sparse matrices to dense in Pytorch. See this series of articles: https://towardsdatascience.com/tagged/predefined-sparsity
- Part 1: Comparing torch.mm to torch.sparse.mm on CPU (2014 Macbook pro)
- Part 2: Comparing torch.mm to torch.sparse.mm on GPU (Tesla V100 on AWS p3.2xlarge)
- Part 3: Comparing sparse mult with dense mult, both using torch.mm, on both CPU and GPU. This is to see if just filling a matrix with 0s and using the same routines as any regular dense matrix provides any speedup. For this part, sparsity is implemented in 2 ways:
  - Randomly filling the matrix with 0s
  - Making the nonzero elements of the matrix have a clash-free structure, as explained in our paper [Pre-Defined Sparse Neural Networks with Hardware Acceleration](https://ieeexplore.ieee.org/document/8689061)
