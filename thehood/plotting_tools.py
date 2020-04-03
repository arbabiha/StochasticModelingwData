"""
Tools for computing and plotting PDFs from sample data.

Used in "Data-driven modeling of strongly nonlinear chaotic systems 
with non-Gaussian statistics" by H. Arbabi and T. Sapsis
April 2019, arbabiha@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from scipy.stats import norm


NiceBlue=cm.Blues(240)

def pdf_1d(f, nx = 100, smoothing_sigma = None, MyRange = None):
    """Computes (a smoothed) PDF of 1D data.
    
    Args:
        f: 1d array of sample data.
        nx: number of bins.
        smoothing_sigma: the STD of a Gaussian kernel convolved with the pdf. 
            If None, no smoothing is done.
        MyRange: the range of the pdf domain. Used when there are outliers
            that mess up the histogram bining.
    
    Returns:
        xx_: the domain of PDF
        rho_: the PDF values on xx_ 
        """


    count, bins  = np.histogram(f, bins=nx, density=True,range=MyRange)
    xx_=0.5*(bins[0:-1]+bins[1:])
    rho_=count

    if smoothing_sigma is not None:
        rho_=np.concatenate((rho_[0]*np.ones((1)),rho_,rho_[-1]*np.ones((1))))
        rho_smoothed = gaussian_filter(rho_,sigma=smoothing_sigma)
        rho_ = rho_smoothed[1:-1]
    return xx_, rho_



def pdf_1d_wCI(f, nx=100, smoothing_sigma=None, MyRange=None):
    """Computes PDF of 1D data and its confidence interval.
    
    To compute the confidence interval, we use "adjusted Wald" 
    in Agresti & Coull 1998.
    
    Args:
        f: 1d array of sample data.
        nx: number of bins.
        smoothing_sigma:the STD of a Gaussian kernel convolved with the pdf.
            If None, no smoothing is done.
        MyRange: the range of the pdf domain. Used when there are outliers
            that mess up the histogram bining.
    
    Returns:
        xx_: the domain of PDF
        rho_: the PDF values on xx_
        rho_u: upper bound of confidence interval for PDF
        rho_l: lower bound of confidence interval for PDF
    """
    
    count, bins  = np.histogram(f, bins=nx, density=False, range=MyRange)
    xx_=0.5*(bins[0:-1]+bins[1:])
    bwidth = bins[1]-bins[0]
    N = f.shape[0]

    # probability of falling in each bin with 
    # extra two successes and two failures

    p=( count+2 ) / ( N  + 4)

    rho_= p/bwidth  # expectation of density
    s_ = np.sqrt(p*(1-p)/N)  / bwidth   # standard deviation of binomial dist
    rho_u,rho_l = rho_+ 2*s_,rho_-2*s_

    if smoothing_sigma is not None:
        rho_= _smoothy(rho_,smoothing_sigma)
        rho_l= _smoothy(rho_l,smoothing_sigma)
        rho_u= _smoothy(rho_u,smoothing_sigma)
        
    return xx_,rho_,rho_u,rho_l

def _smoothy(r, sig):
    """Smoothes signal r by convolving with Gaussian of STD sig."""

    r=np.concatenate((r[0]*np.ones((1)),r,r[-1]*np.ones((1))))
    r_smoothed = gaussian_filter(r,sigma=sig)
    r_smoothed = r_smoothed[1:-1]
    return r_smoothed


def plot_pdf_1d(f,nx = 100, smoothing_sigma = None, **kwargs):
    """Computes and plots the 1D PDF of sample data."""

    xx,rr=pdf_1d(f,nx=nx,smoothing_sigma=smoothing_sigma)
    plt.plot(xx,rr,**kwargs)

def pdf_2d(f, g, nx=100, ny=100, smoothing_sigma=None, MyRange=None):
    """Computes joint PDF of 2D data.
    
    Args:
        f: 1d array of sample data.
        g: 1d array of sample data.
        nx: number of bins for f.
        ny: number of bins for g.
        smoothing_sigma: the STD of a Gaussian kernel convolved with the pdf.
            If None, no smoothing is done.
        MyRange: the range of the pdf domain. Used when there are outliers
            that mess up the histogram bining.
    
    Returns:
        xx_,yy_: the meshgrid domain of PDF
        rho_: the PDF values on xx_,yy_
    """

    rho_,xedges,yedges=np.histogram2d(f, g, bins=[nx,ny], density=True,range=MyRange)
    xx_,yy_ = 0.5*(xedges[0:-1]+xedges[1:]),0.5*(yedges[0:-1]+yedges[1:])

    if smoothing_sigma is not None:
        rho_=gaussian_filter(rho_,sigma=smoothing_sigma)

    return xx_,yy_,rho_




def staircase_plot(f,nx=100,ny=100,variable_name=None,figsize=[10,8],xt=[],yt=[]):
    """Generates a quick plot of 1d and 2d marginals."""
    nrow = np.size(f,1)
    plt.figure(figsize=figsize)
    for j in range(0,nrow):
        nplt = j*nrow + j+1
        xx,rr=pdf_1d(f[:,j],nx=nx)
        ax=plt.subplot(nrow,nrow, nplt)
        ax.plot(xx,rr)
        x1,x2=ax.get_xlim()
        if variable_name is not None:
            title_j=('$'+variable_name+'_'+str(j+1)+'$')
            ax.set_title(title_j)
        for i in range(0,j):
            nplt = j*nrow + i +1
            xx,yy,rho2=pdf_2d(f[:,i],f[:,j],nx=nx,ny=ny)
            ax=plt.subplot(nrow,nrow, nplt)
            plt.contourf(yy,xx,rho2,100)
            plt.set_cmap('jet')

            if variable_name is not None:
                ax.set_title('$'+variable_name+'_'+str(i+1)+'$' + ' vs ' + title_j )
            plt.xticks(xt),plt.yticks(yt)
    plt.subplots_adjust(hspace=.7,wspace=.7)

def staircase_dist(f,nx=100,ny=100,variable_name=None,figsize=[10,8],xt=[],yt=[],Colors='Blues'):
    """Generates a quick plot of 1d and 2d marginals."""
    nrow = np.size(f,1)
    plt.figure(figsize=figsize)
    CMAP=cm.get_cmap(Colors,lut=100)
    LineColor=CMAP(99)
    for j in range(0,nrow):
        nplt = j*nrow + j+1
        xx,rr=pdf_1d(f[:,j],nx=nx)
        ax=plt.subplot(nrow,nrow, nplt)
        ax.plot(xx,rr,color=LineColor)
        x1,x2=ax.get_xlim()
        if variable_name is not None:
            title_j=('$'+variable_name+'_'+str(j+1)+'$')
            ax.set_title(title_j)
        for i in range(0,j):
            nplt = j*nrow + i +1
            xx,yy,rho2=pdf_2d(f[:,j],f[:,i],nx=nx,ny=ny)
            ax=plt.subplot(nrow,nrow, nplt)
            plt.contourf(yy,xx,rho2,100)
            plt.set_cmap(Colors)

            if variable_name is not None:
                ax.set_title('$'+variable_name+'_'+str(i+1)+'$' + ' vs ' + title_j )
            plt.xticks(xt),plt.yticks(yt)
    plt.tight_layout()


def Marginals_plot(f,nx=100,ny=100,Titles=None,figsize=[10,8],Colors='Blues',ticks=[],lims=[],tfs=14):
    """Generates a quick plot of 1d and 2d marginals -- used in the paper."""
    nrow = np.size(f,1)
    myfig=plt.figure(figsize=figsize)
    CMAP=cm.get_cmap(Colors,lut=100)
    LineColor='black'
    for j in range(0,nrow):
        xx,rr=pdf_1d(f[:,j],nx=nx,smoothing_sigma=2,MyRange=lims)
        ax=MarginalAxes(j+1,j+1,myfig,Dim=nrow)
        ax.plot(xx,rr,color=LineColor)
        ax.set_xlim(lims)
        x1,x2=ax.get_xlim()
        plt.yticks,plt.xticks([])
        if Titles is not None:
            plt.title(Titles[j],fontsize=tfs)

        for i in range(0,j):
            xx,yy,rho2=pdf_2d(f[:,j],f[:,i],nx=nx,ny=ny,smoothing_sigma=2,MyRange=[lims,lims])
            ax=MarginalAxes(j+1,i+1,myfig,Dim=nrow)
            plt.contourf(yy,xx,rho2,100)
            plt.set_cmap(Colors)
            plt.xticks([]),plt.yticks([])
            if j==nrow-1:
                plt.xticks(ticks)
            if i==0:
                plt.yticks(ticks)
   


def MarginalAxes(i,j,myfig,Dim=10):
    """A general format for marginal PDFs.
    
    i,j are the row column indices of axes from top left
    """

    h=.92/Dim - .01
    w=.92/Dim - .01
    dw = .01
    dh = .01

    x0,y0=.03,1-7*dh-h

    xl = x0 + (j-1)*(w+dw)
    yl = y0 - (i-1)*(h+dh)

    my_axes = myfig.add_axes([xl,yl,w,h])
    plt.tick_params(direction='in')
    plt.yticks([])
    # plt.xticks([])
    return my_axes  

def MarginalAxes_10d(i,j,myfig):
    """A 10D axes for marginal PDFs.
    
    i,j are the row column indices of axes from top left
    """

    Dim=10
    h=.94/Dim - .01
    w=.94/Dim - .01
    dw = .01
    dh = .01

    x0,y0=.03,1-5*dh-h

    xl = x0 + (j-1)*(w+dw)
    yl = y0 - (i-1)*(h+dh)

    my_axes = myfig.add_axes([xl,yl,w,h])
    plt.tick_params(direction='in')
    plt.yticks([])
    return my_axes  


def Gaussian_fit(y,r=None):
    """Gives a Gaussian PDF fit to the data.
    
    Args:
        y: 1d array of sample data
        r: the grid over which the PDF is computed, if None
            uses 2.5 STD on each side

    Returns:
        r: the grid
        p: Gaussian PDF values on r
    """

    sigma = np.sqrt(np.var(y))
    mu = np.mean(y)

    if r is None:
        r = np.linspace(mu-2.5*sigma,mu+2.5*sigma,num=100)
    
    p = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (r - mu)**2 / (2 * sigma**2) )
    return r,p
