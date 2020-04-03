"""
Tools for analysis and identification of SDE systems.

Used in "Data-driven modeling of strongly nonlinear chaotic systems 
with non-Gaussian statistics" by H. Arbabi and T. Sapsis
A good reference for theory is "Stochastic Differential equations"
by K. Sobczyk
April 2019, arbabiha@gmail.com.
"""


import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.spatial.distance as Distance
from scipy import optimize
import time
import scipy.io as sio
import scipy.signal as signal
from scipy import integrate
import sdeint
from numba import jit
from scipy import interpolate
import timeit
import spectral_analysis as sa
from pyswarm import pso




class StochasticSystem(object):
    """Class of stochastic systems.

    Attributes:
        F (callable): drift term.
        G (callable): diffusion term.
    """

    def __init__(self, F=None, G=None):
        """Initis the class with drift and diffusion."""

        self.F  =  F    
        self.G  =  G    
        
    def GenerateTrajectory(self,y0=np.array([0,0.01]),T=3000,N=30000):
        """Computes a sample trajectory of the system.

        Args:
            y0: initial condition
            T: length of time interavkl for trajectory
            N: number of time steps

        Returns:
            t: array of time stamps
            Y: values of state at t
        """
        
        # t_run = timeit.default_timer()
        t,Y=EM_numba(self.F,self.G,y0,T,N)
        # print('EM numba took '+str(timeit.default_timer() - t_run)+'secs')
        return t,Y

class Stochastic_Oscillator(StochasticSystem):
    """Class of nonlinear stochastic oscillators.

    Attributes:
        pforce (callable): the potential force as a function of displacement
        beta (flaot): the (linear) damping coefficient
        D (float): intensity of white noise forcing
        tau (float): correlation time of systems displacement

    """
    def __init__(self,pforce,beta,D):
        """Constructs the Stochastic_Oscillator class."""
        
        self.pforce = pforce

        # we use another method to set D and beta
        # bc we are going to do it a lot

        self._set_beta_D(beta,D)

    def _set_beta_D(self,beta,D):
        """Sets the free parameters of stochastic oscillator."""

        self.beta=beta
        self.D = D
        SigmaMat=np.array([0,np.sqrt(2*D)])
        pforce = self.pforce # we de-objectify pforce for later jit
        
        @jit
        def Drift(u):
            f1 = u[1]
            f2 = -beta*u[1] - pforce(u[0])
            f = np.array([f1,f2])
            return f
        
        @jit
        def Diffusion(u):
            return SigmaMat

        self.F = Drift
        self.G = Diffusion

    
    def tau_sys(self, beta, D, **trajectory_info):
        """Computes the correlation time of system at parameter values beta and D."""

        self._set_beta_D(beta,D)
        np.random.seed(42)
        t2,Y2=self.GenerateTrajectory(**trajectory_info)
        dt = t2[1]-t2[0]
        tau=CorrelationTime(Y2[0,:],dt=dt)
        self.tau = tau
        print('beta='+str(self.beta)+', D='+str(self.D))
        print('gives tau='+str(self.tau))
        print('============================')
        return tau

    def Match_Correlation(self, tau_target, beta0, alpha, **trajectory_info):
        """Optimizes beta and D to match a target correlation time.
        
        Args:
            tau_target (float): the target value of correlation time.
            beta0 (float): initial guess for beta.
            alpha (float): b/D ratio.
            
        Returns:
            sets the optimal values of beta and D, also returns the sequence of tried
            values for beta and tau.
        """
        
        tau_mismatch = lambda b : (self.tau_sys(b,b/alpha,**trajectory_info) - tau_target)/tau_target

        beta_vals,tau_vals=bisection(tau_mismatch,beta0[0],beta0[1],tol=1e-2,Max_trial=20)
        xn,Fn=Stochastic_RootFinder(tau_mismatch,beta_vals[-1],max_iter=10)

        print('new value of tau ='+str(self.tau))
        return beta_vals,tau_vals 

class Linear_Stochastic_Oscillator(Stochastic_Oscillator):
    """Class of linear stochastic oscillators.

    A subclass of Stochastic_Oscillators with the difference
    that pforce(x):=kx.

    See base class for attributes.
    """

    def __init__(self,k,beta,D):
        """Initis the linear stochastic oscillator class."""
        @jit
        def f1(x):
            return k*x
        self.pforce = f1
        
        # we use another method to set D and beta
        # bc we are going to do it a lot
        self._set_beta_D(beta,D)




def SystemID_spec_match(q,dt=.1):
    """Finds a linear stochastic oscillator with closest PSD to data.
    
    Args:
        q (np.ndarray): n*dim array of time series, each column is a random variable
        dt: the sampling interval
    
    Returns:
        Dictionary of system parameters for dim linear stochastic oscillators
    """

    Dim = q.shape[1]

    k_opt,D_opt = np.zeros(Dim),np.zeros(Dim)

    # loop over dimensions
    for j in range(0,Dim):
        print('identifying system #'+str(j+1)+'via spectral match')
        print(50*'-')
        # first we optimize in q-space
        k_opt[j],D_opt[j]=Optimize_Sqq(q[:,j],dt)

    Sys_Params={'k': k_opt, 'beta':D_opt/k_opt,'D':D_opt}
    
    return Sys_Params

def Optimize_Sqq(qj,dt):
    """Finds the best parameters of a linear stochastic oscillator that match the PSD of data.

    Args:
        qj (np.ndarray): 1d array of time series
        dt (float): sampling interval
    
    Returns:
        k and D parameters of the oscillator

    """

    # find the signal psd
    fs=1.0/dt
    M,L = 256,512
    wxx,Sxx=sa.Welch_estimator(qj,fs=fs,M=M,L=L)
    w_threshold = fs*np.pi/2
    Sxx=Sxx[wxx<w_threshold]
    wxx=wxx[wxx<w_threshold]

    # the psd of oscillator model is computed using Wiener-Khinchin relation
    # and called in the objective function
    print('S_qq matching objective fun: ls error of spectrum')
    spec_ls_distance = lambda params: np.linalg.norm(Sxx*(Sxx - sa.Oscillator_Spectrum(params[0],params[1]/params[0],params[1],wxx)),ord=1)

    # optimize via pyswarm - v1
    lb = [0.001, 0.001]
    ub = [100, 500]
    xopt, fopt = pso(spec_ls_distance, lb, ub, maxiter=10000,swarmsize=10000,minfunc=1e-10)
    k,D = xopt[0],xopt[1]

    b = D/k
    print('result: k='+str(k)+' b='+str(b)+' D='+str(D))

    return k,D

def SystemID_tau_match(q,dt=.1):
    """Finds the best parameters of an oscillator that match the correlation time of data.

    Args:
        q (np.ndarray): n*dim array of time series, each column is a random variable
        dt: the sampling interval
    
    Returns:
        Dictionary of parameters of dim oscillators
    """
    Dim = q.shape[1]

    Sys_Params={'k': np.zeros(Dim), 'beta':np.zeros(Dim),'D':np.zeros(Dim)}

    # loop over dimensions
    for j in range(0,Dim):
        print('identifying system #'+str(j+1)+'via corrleation time match')
        print(50*'-')
        Sys1 = Optimize_tau(q[:,j],dt=dt)
        Sys_Params['k'][j],Sys_Params['beta'][j],Sys_Params['D'][j]=Sys1.k,Sys1.beta,Sys1.D

    return Sys_Params

def Optimize_tau(x,dt):
    """Finds the best parameters of an oscillator that match the correlation time of data.

    Args:
        x (np.ndarray): 1d array of time series, each column is a random variable
        dt: the sampling interval
    
    Returns:
        MySys: a Linear_Stochastic_Oscillator instance with optimal parameters
            MySys has a standard normal pdf as invariant meausre. Also MySys displacement
            has same correlation time as x and its velocity dist. is Guassian 
            and has same variance as xdot.
    """

    xdot = central_diff(x,dt)
    var_xdot = np.var(xdot)
    tau_target = CorrelationTime(x,dt=dt)

    
    k = var_xdot
    beta0 = 0.1
    alpha = 1.0/k  # beta/D ratio

    @jit
    def f1(x):  # spring force
        return k*x
    
    MySys = Stochastic_Oscillator(f1,beta0,beta0 / alpha)
    R = 5000 # how many taus required for integration
    MySys.Match_Correlation(tau_target,np.array([1e-2,20]),alpha,T=R*tau_target,N=int(R*100))
    MySys.k=k

    return MySys




def draw_trajectory_SDEsys(Oscillatior_Parametrs, q0 = np.array([.1,.1]), T = 10000, dt = 0.1):
    """Generates a trajectory of the SDE model.

    Args:
        Oscillatior_Parametrs: dictionary of aparemetrs for stochastic oscillators
        q0: initial condition for each oscillator
        T: length of trajecvtory
        dt: time step
    
    Returns:
        t_model: time stamps of trajectory
        q_model: state values
    """


    N=int(T*1000)   # number of time steps
    n_samp = np.amax([int(dt/(T/N)),1])  # sampling interval (cannot be smaller than time step)


    k,beta,D = Oscillatior_Parametrs['k'],Oscillatior_Parametrs['beta'],Oscillatior_Parametrs['D']

    Dim = len(k)
    q_model = np.zeros((N+1,Dim))

    for j in range(0,Dim):
        MySys=Linear_Stochastic_Oscillator(k[j],beta[j],D[j])
        t4,X4=MySys.GenerateTrajectory(q0,T=T,N=N)
        q_model[:,j]=X4[0,:]

    # subsample and push out
    q_model,t_model =q_model[::n_samp,:],t4[::n_samp]

    return t_model,q_model



@jit
def EM_numba(F,G,Y0,T=100,N=500000):
    """Euler-Maruyama method for integarting a trajector of SDE.

    Args:
        F (callable): the drift term
        G (callable): the diffusion term
        Y0 (array or list): initial condition
        T: length of trajectory
        N: number of time steps

    Returns:
        t: array of trajectory time stamps
        Y: state values on trajectory
    """

    Y0 = np.array(Y0)
    Y = np.zeros((Y0.shape[0],N+1))
    Y[:,0]=Y0
    dt = T/N
    t=np.linspace(0,T,N+1)

    for jt in range(0,N):
        Y[:,jt+1]=_EM_step(F,G,Y[:,jt],dt)

    return t,Y

@jit
def _EM_step(F,G,Y,dt):
    """Takes one step of Euler-Maruyama method."""
    dW = np.sqrt(dt) * np.random.randn(Y.shape[0])
    Ynext = Y + F(Y) * dt + G(Y) * dW
    return Ynext



def CorrelationTime(y,dt=1):
    """Computes the correlation time of time series in y."""

    ac=np.correlate(y,y,'same')
    Index_m= int( (len(y)-1) / 2)   # index of middle point (lag=0)
    Slab=ac[Index_m:]               # take the right half
    ID_zero= np.argmax(Slab<0)-1    # first crossing with zero
    n_corr = np.sum(Slab[:ID_zero])/Slab[0]
    tau_corr=n_corr*dt
    return tau_corr

def bisection(f,a,b,tol=1e-4,Max_trial=14):
    """Solves f(p)=0 with p in [a,b] using bisection method.
    
    The coding is optimized so that f is computed the minimal number
    of required times.

    Args:
        f (callable): the function f in f(p)=0
        a (float): lower bound of admissible solution
        b (float): upper bound of admissible solution
        tol: tolerance of solution, i.e., required |f(x)| for convergence
        Max_trial: maximum number of trying (evaluating+1) f

    Returns:
        pvalues: the tried values for p
        fvalues: corresponding values for f(p)
    
    """
    pvalues = np.zeros(Max_trial+3)
    fvalues = np.zeros(Max_trial+3)
    fa,fb = f(a),f(b)
    pvalues[0],pvalues[1]=a,b
    fvalues[0],fvalues[1]=fa,fb

    if fa*fb>0:
        print('f(a)*f(b)>0 ---> specified a and b do not work!')
        print('spiting out a or b --- whichever is closer')
        
        if np.abs(fa)<=fb:
            return a,f(a)
        else:
            return b,f(b)
            
        
    p=(a+b)/2
    pvalues[2]=p
    fp=f(p)
    fvalues[2]=fp
    err=np.abs(fp)

    j=0
    while err > tol and j<Max_trial:

        if fa*fp<0:
            b=p
            p=(a+b)/2
            fp = f(p)
        else:
            a=p
            fa = fp
            p=(a+b)/2
            fp = f(p)

        err=abs(fp)
        print('-------------------------------------------------------')
        print('bisection trial #'+str(j)+':  sol='+str(p)+', mismatch ='+str(err))
        print('-------------------------------------------------------')
        j=j+1
        pvalues[j+2]=p
        fvalues[j+2]=fp
        # print('p='+str(p))
        # print('a,b='+str(a)+'--'+str(b))

    pvalues,fvalues=pvalues[:j+3],fvalues[:j+3]

    return pvalues,fvalues

def Stochastic_RootFinder(F, x0, a=.5, alpha=0.5, max_iter=20, xtol=.01, ValidInterval=[0.00001,100]):
    """Solving f(x)=0 when we can only evaluate f with some noise.
    
    We use the suggested algorithm in 
    # "The Stochastic Root Finding Problem: Overview, 
    # Solutions, and Open Questions" by Pasupathy & Kim 2011
    
    Args:
        F (callable): an unbiased estimator of f
        x0 (float): first guess for x
        ValidInterval: the search interval
        max_iter: maximum number of allowed iterations
        To see other params check out the above paper.

    Returns:
        the final iteration value of x and F(x)
    """

    xp,n= ValidInterval[1]*2,1
    n0 = int(max_iter/20)+1
    xn = x0
    displacement = 10 * xtol

    while displacement > xtol and n<max_iter:
        an = a/ ( (n+n0)** alpha)
        Fn=F(xn)
        print('SRF step '+ str(n))
        print('--------------------------')
        print('current value of x is '+ str(xn))
        print('current value of F is '+ str(Fn))
        xnext = xn-an*np.asscalar(Fn)
        
        if xnext>ValidInterval[1]:
            xnext = ValidInterval[1]
        elif xnext<ValidInterval[0]:
            xnext = ValidInterval[0]
        
        print('new x is ' + str(xnext))



        # shift and compute total displacement in the last 2 steps
        displacement = np.ptp(np.array([xp,xn,xnext]))
        print('displacement in the last 3 steps is ' + str(displacement))
        xp=xn
        xn=xnext
        

        n=n+1

    return xnext,Fn

def central_diff(v,dx=1):
    """Simple finite difference for 1st derivative."""
    
    dv = ( v[2:]-v[0:-2] ) / (2.0*dx)
    return dv