"""
Tools for approximation of Koopman continuous spectrum (aka PSD).

Used in "Data-driven modeling of strongly nonlinear chaotic systems with non-Gaussian statistics" 
by H. Arbabi and T. Sapsis
April 2019, arbabiha@gmail.com
"""

import numpy as np
import math
import scipy.io as sio
import scipy.optimize
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numba import jit

import SDE_systems as sm

def Welch_estimator(x, M=100, L=200, fs=1):
    """Computes the power spectral density via Welch method.

    Based on the paper by P. Welch
    'The use of fast Fourier transform for the estimation of 
    power spectra: A method based on time averaging over short, 
    modified periodograms', IEEE Trans. Audio Electroacoust. 15, 70 (1967).

    Args:
     x: 1d arrayt of signal 
     M: length of each block used     
     L: size of the padded FFT grid
     fs: sampling frequency (in Hz)

    Returns:
        omega: array of frequency grid (in rad/sec)
        phi: the power spectral density over omega
    """

    K = int(M/2)    # overlap length
    N = x.shape[0]
    v = np.blackman(M)/10    # the window function

    S = int(math.floor(((N-M+K)/K)))

    Pv=np.mean(v**2)

    # function handle to compute psd for each block
    psd_fft = lambda y: (np.abs(np.fft.fft(  (y[0:M]*v),n=L) )**2 ) /M


    phi = np.zeros(L)

    for j in range(0,S):
        phi = phi + psd_fft(x[j*K:j*K+M] )
    
    phi = phi/(S*Pv*fs )
    
    # the freqency grid
    omega = 2*np.pi*fs*np.linspace(0,1,L)

    return omega,phi




def approx_PSD_w_delta(wg,p,nw=100,fs=1):
    """Approximates the PSD with delta functions.
    
    This is equivalent to approximating a chaotic signal
    with quasi-periodic signa.

    Args:
        wg (np.ndarray): array of frequency grid
        p (np.ndarray): PSD values corresponding to wg
    
    Returns:
        w: array of discrete frequncy values
        a: array of amplitudes (sqrt of power) for discrete frequencies
    """

    # generate a random set of intervals on [0,pi]
    e =np.append(np.random.rand(nw-1)*np.pi*fs,[0,np.pi*fs])
    e = np.sort(e) # endpoints of intervals
    dw = np.diff(e)  # length of intervals
    w = (e[:-1]+e[1:])/2    # midpoint of intervals

    rho_fun = interp1d(wg,p/(2*np.pi),kind='linear')
    rho= rho_fun(w)

    # amplitude of each delta
    a = np.sqrt(2*2*rho*dw)
    # first 2 is to account for the pi<w<2pi
    # the second 2 is to move all the energy into the real part
    
    # psd_measure = np.trapz(p,wg)
    # qp_measure = np.sum(a**2)

    return w,a,dw,rho


def Oscillator_Spectrum(k,b,D,w):
    """Computes the response PSD of a linear stochastic oscillator.
    
    The oscillator given by
    kx + b xdot + xddot = sqrt{2D}Wdot
    Its PSD is computed using Wiener-Khinchin relation.

    Args:
        k (float): stiffness
        b (float): damping coeff
        D (float): intensity of white noise forcing
        w (np.ndarray): 1d array of frequencies on which the PSD of x is evaluated

    Returns:
        array of PSD values on w
    """
    # returns the response PSD of a stochastic oscillator
    # with stiffness k, damping coeff b, and wgn forcing amplitude sqrt{2D}
    # at grid values of frequency w (rad/sec)
    # using Wiener-Khinchin relation

    # transfer function
    H = lambda omega: 1.0/(k-omega**2+ 1j *b*omega)

    rho = 2*D* (np.abs(H(w))**2 )

    return rho


def test_Welch_vs_WKrelation():
    """Verifies that PSD computed by Welch matches the analytical formula."""


    # form a linear system (damper and spring)
    k,b,D = 2.91,1.58,4.60
    MySys = sm.Linear_Stochastic_Oscillator(k,b,D)

    T=10000
    N=int(T*100)
    t4,X4=MySys.GenerateTrajectory(np.array([.1,.1]),T=T,N=N)

    # subsample and spit out
    njump=10
    t,y = t4[::njump],X4[0,::njump]

    fs = 1.0/(t[1]-t[0])

    # compute psd
    M,L = 512,1024
    omega,phi = Welch_estimator(y,fs=fs,M=M,L=L)

    plt.figure()
    plt.plot(omega,phi,label=r'$f_s=$'+str(fs))


    # analytic solution
    plt.plot(omega,Oscillator_Spectrum(k,b,D,omega),'--k',label=r'$2D/|H(\omega)|^2$')
    plt.legend()
    plt.xlim(0,5)
    plt.title(r'Wiener-Khinchin relation $k=$'+str(k)+', $b=$'+str(b)+', $D=$'+str(D))
    plt.savefig('spectest.png',dpi=400)

def next_power_of_2(x):
    s=int(2**(x.bit_length()))
    return s


if __name__ == '__main__':
    print('testing examples related to spectral analysis ...')
    test_Welch_vs_WKrelation()
    pass

    



