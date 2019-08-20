"""
tools related to analysis and identification of SDE systems used in
"Data-driven modeling of strongly nonlinear chaotic systems with non-Gaussian statistics" 
by H. Arbabi and T. Sapsis
June 2019, arbabi@mit.edu

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





#########################################################################
# class of stochastic models
class StochasticSystem(object):
    def __init__(self,F=None,G=None):

        self.F  =  F    # drift
        self.G  =  G    # diffusion
        
    def GenerateTrajectory(self,y0=np.array([0,0.01]),T=3000,N=30000):
        # we solve the SDE using Euler-Maruyama 
        t_run = timeit.default_timer()
        t,Y=EM_numba(self.F,self.G,y0,T,N)
        print('EM numba took '+str(timeit.default_timer() - t_run)+'secs')
        return t,Y

class Stochastic_Oscillator(StochasticSystem):
    def __init__(self,pforce,beta,D):
        # potential force
        self.pforce = pforce
        # we use another method to set D and beta
        # bc we are going to do it a lot
        self._set_beta_D(beta,D)

    def _set_beta_D(self,beta,D):
        print('setting the parameters of stochastic oscillator')
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

    
    def tau_sys(self,beta,D,**trajectory_info):
        # computing the correlation time of the system
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

    def Match_Correlation(self,tau_target,beta0,alpha,**trajectory_info):
        print('matching the correlation time:')
        
        tau_mismatch = lambda b : (self.tau_sys(b,b/alpha,**trajectory_info) - tau_target)/tau_target

        beta_vals,tau_vals=bisection(tau_mismatch,beta0[0],beta0[1],tol=1e-2,Max_trial=20)
        xn,Fn=Stochastic_RootFinder(tau_mismatch,beta_vals[-1],max_iter=10)

        print('new value of tau ='+str(self.tau))
        return beta_vals,tau_vals 

class Linear_Stochastic_Oscillator(Stochastic_Oscillator):
    def __init__(self,k,beta,D):
        # potential force
        @jit
        def f1(x):
            return k*x
        self.pforce = f1
        # we use another method to set D and beta
        # bc we are going to do it a lot
        self._set_beta_D(beta,D)

########################################################################
# some routines for system ID 

def SystemID_spec_match(q,dt=.1):
    # this program identifies a set of stochastic oscillators 
    # that produce the closest power spectral density to the signals
    #  stored in columns of q  with sampling interval dt
    Dim = q.shape[1]

    k_opt,D_opt = np.zeros(Dim),np.zeros(Dim)

    # loop over dimensions
    for j in range(0,Dim):
        print('identifying system #'+str(j+1)+'via spectral match')
        print(50*'-')
        # first we optimize in q-space
        k_opt[j],D_opt[j]=Optimize_Sqq(q[:,j],dt)

    # return optimal parameters
    Sys_Params={'k': k_opt, 'beta':D_opt/k_opt,'D':D_opt}
    
    return Sys_Params

def Optimize_Sqq(qj,dt):
    # this function finds the optimal values of (k,D) for a linear oscillator
    # so that the PSD of its response to WGN is closest to PSD of time series q_j 
    # with sampling interval dt
    

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
    # this program identifies a set of stochastic oscillators 
    # that produce the closest correlation time to the signals
    #  stored in columns of q  with sampling interval dt
    # it also matches the variance of \dot{q}
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
    # inputs: x scalar time series (with standard normal distribution)
    #         dt sampling interval of x
    # outputs: MySys a stochastic oscillator object 
    #           that has standard normal pdf as invariant meausre
    #           and MySys displacement has same correlation time as x
    #           and MySys velocity is Guassian and has same variance as xdot

    xdot = central_diff(x,dt)
    var_xdot = np.var(xdot)
    tau_target = CorrelationTime(x,dt=dt)

    # initialize a system
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

    # return the oscillator object
    return MySys




def draw_trajectory_SDEsys(Oscillatior_Parametrs,q0=np.array([.1,.1]),T=10000,dt=0.1):
    # generate a trajectory of the SDE model
    # given system parameters in q space

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




########################################################################
# Euler-Maruyama integration via numba
@jit
def EM_numba(F,G,Y0,T=100,N=500000):

    Y0 = np.array(Y0)
    Y = np.zeros((Y0.shape[0],N+1))
    Y[:,0]=Y0
    dt = T/N
    t=np.linspace(0,T,N+1)

    for jt in range(0,N):
        Y[:,jt+1]=EM_step(F,G,Y[:,jt],dt)

    return t,Y

@jit
def EM_step(F,G,Y,dt):
    dW = np.sqrt(dt) * np.random.randn(Y.shape[0])
    Ynext = Y + F(Y) * dt + G(Y) * dW
    return Ynext


########################################################################
# some auxiliary functions

def CorrelationTime(y,dt=1):
    # compute the correlation time of time series
    ac=np.correlate(y,y,'same')
    Index_m= int( (len(y)-1) / 2)   # index of middle point (lag=0)
    Slab=ac[Index_m:]               # take the right half
    ID_zero= np.argmax(Slab<0)-1    # first crossing with zero
    n_corr = np.sum(Slab[:ID_zero])/Slab[0]
    tau_corr=n_corr*dt
    return tau_corr

def bisection(f,a,b,tol=1e-4,Max_trial=14):
    # solving f(x)=0 with x in [a,b] 
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

def Stochastic_RootFinder(F,x0,a=.5,alpha=0.5,max_iter=20,xtol=.01,ValidInterval=[0.00001,100]):
    # solving f(x)=0
    # while we have access to a stochastic consistent estimator of f(x)
    # given by F(x)
    # x0, first guess and the rest are parameters
    # following Pasupathy & Kim 2011
    # "The Stochastic Root Finding Problem: Overview, 
    # Solutions, and Open Questions"
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
    dv = ( v[2:]-v[0:-2] ) / (2.0*dx)
    return dv