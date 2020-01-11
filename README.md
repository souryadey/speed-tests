# speed-tests

### This is a repository to test and compare speed and complexity of different programming methods and frameworks.

computation_frameworks -- Compare numpy, torch and tensorflow and Python's native lists

<br>

torch.sparse_&lt;platform&gt;: Characterize the `torch.sparse` API in Pytorch and compare it to using regular dense matrices:
- Part 1: Comparing on CPU (2014 Macbook pro with 16GB RAM). [Blog article](https://towardsdatascience.com/sparse-matrices-in-pytorch-be8ecaccae6)
- Part 2: Comparing on GPU (Tesla V100 on AWS p3.2xlarge). [Blog article](https://towardsdatascience.com/sparse-matrices-in-pytorch-part-2-gpus-fd9cc0725b71)<br>

**For conclusions, see the blog articles.**

<br>

sparsemats_denseroutines_&lt;framework&gt;: This compares _storage requirement_ and _time to multiply matrices_ of different levels of sparsity. Here, **sparse matrices are expressed as regular dense matrices** (i.e. we use full size matrices filled with zeros instead of some special sparse API, which is not very effective in its current state). The sparse matrices are formed from 2 processes:
- Randomly filling with zeros
- Structured according to clash-free pre-defined sparsity, which is [part of my PhD research](https://ieeexplore.ieee.org/document/8689061).<br>

Platforms used are same as above, i.e. 2014 Macbook pro CPU and AWS p3.2xlarge GPU. Frameworks used are:
- Pytorch 1.3.0
- Tensorflow 2.1.0

The time complexity results are given as curves of varying density (percentage of non-zero elements). Mat1 and Mat2 are the 2 sparse matrices being multiplied. For example, the filename `random_100,100x100,100_cpu` implies multiplying 2 randomly sparse matrices of size 100x100 each on CPU. `reversedorder` implies that the densities were swept from high to low instead of low to high. This helps to eliminate any bias due to the system heating up and slowing down as a result.<br>

**Conclusions: Both storage and time complexity do not change as density (or sparsity) is varied.** This is because even though the matrices have a lot of zeros, they are stored and operated on in the same way as a regular dense matrix. (Note that some results might consistently show a larger time taken when density is higher, but the difference is too small to be statistically significant).
