import numpy as np


def memmap_sweep(p,z, typ=3):
    ''' Generate all left memory addresses in a SINGLE sweep. For definitions, see wt_interleaver '''
    if typ==1:
        return np.asarray([np.random.permutation(p//z) for _ in range(z)]).T.reshape(p,)
    elif typ==2 or typ==3:
        s = np.random.choice(np.arange(p//z),z) #pick starting addresses of z memories as any number between 0 and p//z-1 (since #rows = p//z)
        return np.asarray([(s+i)%(p//z) for i in range(p//z)]).reshape(p,)

def memdither_sweep(p=40,z=10, memdith=0):
    ''' Generate vector for memory dither in a SINGLE sweep. For definitions, see wt_interleaver '''
    if memdith==0:
        return np.tile(np.arange(z), p//z) #no mem dither, just read mems in order. Make v the same size as t (p elements) for vectorized calculation of wt_interleaver
    elif memdith==1:
        return np.tile(np.random.permutation(z), p//z) #memory dither, read mems in some other order (held constant for all cycles in a sweep)


def wt_interleaver(p,fo,z, typ=3, memdith=0, deinter=None, inp=None):
    '''
    p: Number of neurons in left layer of junction
    fo: Fanout degree of left layer
    z: Degree of parallelism
    typ:
        Type 1: No restrictions
        Type 2: Subsequent addresses in each actmem are +1 modulo
        Type 3: Type 2, and memmap remains same for every sweep
    memdith:
        Introduce additional memory dither v for every sweep. Eg for z=10 :
            For memdith=0, memories would be accessed as [0123456789] in every sweep
            For memdith=1, 1st sweep may access memories as [5803926174], 2nd sweep as [8279306145]
        Does not come into effect for typ 3
    Returns:
        Interleaver pattern
        De-interleaver pattern if deinter!=None
        Interleaver applied to particular input 'inp' if inp!=None
    '''
    assert (p/z)%1 == 0, 'p/z = {0}/{1} = {2} must be an integer'.format(p,z,p/z)

    ## Initial sweep ##
    t = memmap_sweep(p,z,typ)
    v = memdither_sweep(p,z,memdith)
    inter = (t*z+v)*fo

    ## Following sweeps ##
    for i in range(1,fo):
        if typ!=3:
            t = memmap_sweep(p,z,typ) #generate new t
            if memdith!=0:
                v = memdither_sweep(p,z,memdith) #generate new v
        inter = np.append(inter, (t*z+v)*fo + i)

    assert set(range(p*fo)) == set(inter), 'Interleaver is not valid:\n{0}'.format(inter)
    inter = inter.astype('int32')

    if deinter!=None:
        deinter = np.asarray([np.where(inter==i)[0] for i in range(len(inter))])
        deinter = deinter.astype('int32')

    if inp!=None:
        assert len(inp)==p*fo, 'Input size = {0} must be equal to p*fo = {1}'.format(len(inp),p*fo)
        inp = inp[inter]
        inp = inp.astype('int32')

    return (inter, deinter, inp)


def inter_to_adjmat(inter,p,fo,n):
    ''' Convert a weight interleaver 'inter' to an adjacency matrix '''
    fi = p*fo//n
    adjmat = np.zeros((n,p))
    for i in range(len(inter)):
        adjmat[i//fi,inter[i]//fo] = 1
    return adjmat


def adjmat_clash_free(p,fo,n, z, typ=3, memdith=0):
    '''
    Generates weight interleaver and converts to adjmat
    These are GUARANTEED to be valid and clash-free, no need to check
    If check is still desired, set check=1:
        For clash freedom, cannot look at adjmat because an integral number of rows will NOT be read every cycle in the general case where z is not an integral multiple of fi
        The only way to check clash-freedom is to use the interleaver pattern to look at the left neurons read every cycle and check that they all come from different memories
    '''
    inter,_,_ = wt_interleaver(p,fo,z, typ, memdith)
    adjmat = inter_to_adjmat(inter, p,fo,n)
    return adjmat