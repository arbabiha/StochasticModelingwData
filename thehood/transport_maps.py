"""
a wrapper for computing the transport maps in
"Data-driven modeling of strongly nonlinear chaotic systems with non-Gaussian statistics" 
by H. Arbabi and T. Sapsis
April 2019, arbabi@mit.edu

The transport maps are computed using the TransportMaps package by UQgroup@mit:
https://transportmaps.mit.edu/ 
The methodology is described in
"Transport map accelerated markov chain monte carlo", SIAM/ASA Journal on Uncertainty Quantication
2018, by M. Parno and Y. Marzouk

"""
import TransportMaps as TM
import TransportMaps.Functionals as FUNC
import TransportMaps.Maps as MAPS
import TransportMaps.Distributions as DIST
import SpectralToolbox.Spectral1D as S1D
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import timeit
import logging
TM.setLogLevel(logging.INFO)



def compute_transport_map(x0,polynomial_order=2,MPIsetup=None):
    # x0:               the data matrix (each column represents a random variable)
    # polynomial_order: order of polynomial used
    # MPIsetup:         a list that describes how many cores hould be used
    # for computation of dimensions starting from the last 
    # for example MPIsetup =[2,3,4]
    # uses 4 cores for the n-th dimension, 3 for n-1-th and 2 for n-2th

    # returns the transport map SL 



    Dim = x0.shape[1]

    # create the distribution object for my samples    
    pi = DistributionFromSamples(x0)
    
    # create reference distribution
    rho = DIST.StandardNormalDistribution(Dim)


    # first we pack the distribution into neighborhood of origin 
    # linear adjustment
    beta = 1.0   
    b= beta/np.std(x0,0)
    a= - b * np.mean(x0,0)  # centering
    L = MAPS.FrozenLinearDiagonalTransportMap(a,b)



    # Square Integral formulation of transport maps (robust and fast)
    S = TM.Default_IsotropicIntegratedSquaredTriangularTransportMap(Dim, polynomial_order, 'total')
    print(30*"-")
    print("Computing transport maps with polynomial order: %d" % polynomial_order)
    print("Number of coefficients: %d" % S.n_coeffs)
    

    push_L_pi = DIST.PushForwardTransportMapDistribution(L, pi)
    push_SL_pi = DIST.PushForwardTransportMapDistribution(S, push_L_pi)
    qtype = 0      # Monte-Carlo quadratures from pi
    qparams = np.size(x0,0)  # Number of MC points = all available points
    reg = None     # No regularization
    tol = 1e-10     # Optimization tolerance
    ders = 2       # Use gradient and Hessian

    # MPI setup
    if MPIsetup is None:
        MyPool = Dim*[None]
    else:
        npools=len(MPIsetup)
        MyPool= (Dim-npools)*[None] 

        pool_dic={}
        for jp in range(0,npools):
            pool_dic["mpi_pool{0}".format(jp+1)]=TM.get_mpi_pool()
            pool_dic["mpi_pool{0}".format(jp+1)].start(MPIsetup[jp])
            MyPool.append(pool_dic["mpi_pool{0}".format(jp+1)])
    # print('MyPool for parallization is '+str(MyPool))
    print('Number of cores for parallel dimensions is '+ str(MPIsetup))

    # compute the polynomial map 
    log = push_SL_pi.minimize_kl_divergence(rho, qtype=qtype, qparams=qparams, regularization=reg, tol=tol, ders=ders,maxit=300,mpi_pool=MyPool)

    # compose with the linear part 
    SL = MAPS.CompositeMap(S,L)

    return SL



# a modification of distirbution object from TM 
class DistributionFromSamples(DIST.Distribution):
    def __init__(self, x):
        super(DistributionFromSamples,self).__init__(np.size(x,1))
        self.x = x
    def quadrature(self, qtype, qparams, *args, **kwargs):
        if qtype == 0: # Monte-Carlo
            nmax = self.x.shape[0]
            if qparams> nmax:
                raise ValueError("Maximum sample size (%d) exceeded" % nmax)
            p = self.x[0:qparams,:]
            w = np.ones(qparams)/float(qparams) 
        else: raise ValueError("Quadrature not defined")
        return (p, w)


# computing the inverse of transport maps
def TMinverse(q_samp,T,num_core = 20):
    # q_samp: the data matrix for the variables with the reference measure (each column is a random variable)
    # T: the forward transport map, i.e. q=T(x)
    # num_core: number of cores used in parallel computation of x

    # returns x_back= T^{-1}(q_samp)   

    print('parallel computing the inverse map ...')
    print('data shape='+str(q_samp.shape))
    t_sim = timeit.default_timer()   
    print('num_cores_max ='+str(multiprocessing.cpu_count()))
    print('num_cores being used ='+str(num_core))
    n_chunk = int(q_samp.shape[0]/num_core)
    print('size of chunks='+str(n_chunk))
    inputs = range(num_core) 

    def ComputeTinv(i):
        q_chunk = q_samp[i*n_chunk:(i+1)*n_chunk,:]
        x_chunk = T.inverse(q_chunk)
        return x_chunk


    ParallelChunks = Parallel(n_jobs=num_core, verbose=0)(delayed(ComputeTinv)(i) for i in inputs)


    def back_2_array(Result):
        # putting back the result of parallel computation
        X = Result[0]
        for j in range(1,len(Result)):
            X = np.concatenate((X,Result[j]),axis=0)
        return X
    
    x_back=back_2_array(ParallelChunks)

    print('... took {} seconds'.format(timeit.default_timer() - t_sim))

    return x_back

def Generate_SND_sample(Dim,n=1000000):
    rho = DIST.StandardNormalDistribution(Dim)
    x, w = rho.quadrature(0, n)
    return x