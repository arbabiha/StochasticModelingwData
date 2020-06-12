"""
Stochastic modeling of climate data and tail extrapolation.

An example from "Data-driven modeling of strongly nonlinear chaotic systems with non-Gaussian statistics" 
by H. Arbabi and T. Sapsis
April 2019, arbabiha@gmail.com
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
import scipy.stats as stats
import scipy.optimize
import logging
import dill
import timeit
from joblib import Parallel, delayed
import multiprocessing

# custom tools
from sys import path
path.append('./thehood/')
import plotting_tools as pt
import spectral_analysis as sa
import transport_maps as tm
import SDE_systems as sm


def ExtrapolateTails(idx_target = 0, years_train = 2, polynomial_order = 2,
                     SavePath='./results/'):
    """Extrapolating the tails of climate tim-series.


    We load the time series data from NorthPole_data.mat. Here is its structure:
    y(:,1:3) the time series of respectively U, V and T wavelet 
       coefficients on top of NorthPole (target variables)
    y(:,4:30) the time series of covariates most correlated with U at North Pole
    y(:,58:84) the time series of covariates most correlated with T at North Pole
    npc=27 is the number of covariates for each target variable
    yp is the periodic part of y & yc is the chaotic part of y
    such that y=yp+yc


    Args:
        idx_target: variable to be modeled; 0 is U-velocity, 1 V-velocity and 2 Temperature
        years_train: length of training data in years
        polynomial_order: order of polynomial used for transport map
        SavePath: where to save the data

    
    Returns:
        saves the pulled back data
    
    """
    start_year = 0 # when does the training data start


    Tag = 'NorthPole_var'+str(idx_target)+'_r'+str(polynomial_order)+'_tyrs'+str(years_train)+'_syrs'+str(start_year)
    print('simulation tag: '+Tag)
    num_core = 20  # cores for parallel computing of inverse maps
    ntrials = 5 # max number of covariates used
    nsamp = int(74*365*4) # number of samples for extrapolation 

    # data matters
    TimeSeriesData=sio.loadmat('./thehood/NorthPole_data.mat')
    Y,n_covar = TimeSeriesData['yc'],TimeSeriesData['npc'][0,0]-1
    idx_train = int(years_train*365*4) 
    idx_start = int(start_year*365*4) 
    Y_train = Y[idx_start:idx_start+idx_train,:]



    # variable matters
    idx_help = 3 + np.arange(int(idx_target*n_covar),int((idx_target+1)*n_covar))
    yn_train=Y_train[:,idx_target]
    yn_truth=Y[:,idx_target]

    
    if not os.path.exists(SavePath):
        print('create path ...')
        os.makedirs(SavePath)

    # now choose a set of variables and do extrapolation
    yns_model = np.zeros((nsamp,ntrials))

    for dim in range(0,ntrials):
        print(20*'=')
        print('coupling with '+str(dim) +' variables')
        y = np.concatenate((Y_train[:,idx_target:idx_target+1],Y_train[:,idx_help[0:dim]]) ,axis=1)
        y = np.fliplr(y) # bc we want the coupling to improve y1
        
        # compute the transportmap
        MPIsetup=None
        # if dim>3:  # if MPI of transportmaps is installed
        #     MPItest=[2,2,4,4]
        S =tm.compute_transport_map(y,polynomial_order=polynomial_order,MPIsetup=MPIsetup)
        
        # generate a big sample
        q = tm.Generate_SND_sample(y.shape[1],n=nsamp)
        
        y_model = tm.TMinverse(q,S,num_core=num_core)
        yn_model=y_model[:,-1]
        
        # save ymodel to a larger data-set
        yns_model[:,dim]=yn_model

        # we do the saving in each loop
        sio.savemat(SavePath+Tag,{'y_truth':yn_truth,'y_train':yn_train,'ys_model':yns_model,'n_completed':dim})

    return Tag
    
def Plot_NorthPole_signals(savepath,tag='',picformat='png'):
    """Plots the time series of U-velocity and temperature wavelet coeffs on Northpole."""

    plt.rc('font', family='serif',size=9)
    tfs = 10

    MyColors=['#377eb8','#d95f02']
    lw2=.8

    # data matters
    BaseData=sio.loadmat('./thehood/NorthPole_data.mat')
    t,y,yp,yc=np.squeeze(BaseData['t']),BaseData['y'],BaseData['yp'],BaseData['yc']

    # smoothing for pdfs?
    F=2

    # axes sizes
    myfig=plt.figure(figsize=[6.5,2.4])
    w1,w2,w3=.29,.29,.16
    h=.3
    dh,dw=.11,.07
    x0,y0 = .08,.89-h

    xlims=[1980,1995]

    # u-vel time series
    ax1 = myfig.add_axes([x0,y0,w1,h])
    ax1.tick_params(direction='in')
    ax1.plot(t,y[:,0],color=MyColors[0])
    plt.plot(t,yp[:,0],'--',label=r'periodic part',color=MyColors[1])
    plt.xlim(xlims),plt.ylim(-10,25)
    ax1.set_xticklabels([])
    plt.title(r'time series',fontsize=tfs)
    plt.ylabel(r'$U_{NP}$~(m/s)')
    ax1.yaxis.set_label_coords(-.17,.5)

    legend=plt.legend(fontsize=tfs-2,bbox_to_anchor=(.38, .14),loc='center left',ncol=1,fancybox=True,framealpha=0.0)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_edgecolor("black")




    ax2 = myfig.add_axes([x0+w1+dw,y0,w2,h])
    ax2.tick_params(direction='in')
    ax2.plot(t,yc[:,0],color=MyColors[0])
    plt.xlim(xlims)
    ax2.set_xticklabels([])
    plt.title(r'chaotic part',fontsize=tfs)

    ax3 = myfig.add_axes([x0+w1+w2+2*dw+.02,y0,w3,h])
    ax3.tick_params(direction='in')
    [rr,pr]=pt.pdf_1d(yc[:,0],nx=100,smoothing_sigma=F)
    rg,pg=pt.Gaussian_fit(yc[:,0],r=rr)
    plt.plot(rg,pg,'k--',label=r'Gaussian fit',linewidth=lw2)
    plt.plot(rr,pr,color=MyColors[0])
    ax3.set_yscale('log')
    plt.ylim(1e-6,.5)
    plt.yticks([1e-1,1e-3,1e-5])
    plt.xlim(-12,12),plt.xticks([-10,10])


    plt.title(r'PDF of chaotic part',fontsize=tfs)
    legend=plt.legend(fontsize=tfs-2,bbox_to_anchor=(-.05, -1.75),loc='center left',ncol=1,fancybox=True,framealpha=0)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_edgecolor("black")

    # temperature time series
    ax4 = myfig.add_axes([x0,y0-h-dh,w1,h])
    plt.tick_params(direction='in')
    plt.plot(t, y[:,2])
    plt.plot(t,yp[:,2],'--',color='#d95f02')
    plt.xlim(xlims)
    plt.xlabel(r'$t$ (year)')
    plt.ylabel(r'$T_{NP}$~(K)')
    plt.ylim(135,155)

    ax5 = myfig.add_axes([x0+w1+dw,y0-h-dh,w2,h])
    ax5.tick_params(direction='in')
    ax5.plot(t,yc[:,2])
    plt.xlim(xlims)
    plt.xlabel(r'$t$ (year)')

    ax6 = myfig.add_axes([x0+w1+w2+2*dw+.02,y0-h-dh,w3,h])
    ax6.tick_params(direction='in')
    [rr,pr]=pt.pdf_1d(yc[:,2],nx=100,smoothing_sigma=F)
    rg,pg=pt.Gaussian_fit(yc[:,2],r=rr)
    plt.plot(rg,pg,'k--',linewidth=lw2)
    plt.plot(rr,pr,color=MyColors[0])
    ax6.set_yscale('log')
    plt.yticks([1e-1,1e-3,1e-5]),plt.ylim(1e-6,.8)
    plt.xlim(-10,10),plt.xticks([-5,5])

    plt.savefig(savepath+'NorthPole_UT.'+picformat,dpi=400)



def Plot_tails_wCI(savepath,tag='',picformat='png'):
    """Plots the extrapolated tails U & T wavelet coeffs on Northpole."""

    plt.rc('font', family='serif',size=9)
    tfs = 10

    MyColors=['#377eb8','#d95f02']
    
    lw0,lw1,lw2=1.5,1.2,1.2

    # smoothing for pdfs?
    F=2 #std of gaussian kernel in convolution 

    # axes sizes
    myfig=plt.figure(figsize=[6.5,2.4])
    w=.16
    h=.3
    dh,dw=.11,.02
    x0,y0 = .11,.89-h

    # data tags
    Setup,years_train,polynomial_order,start_year = 21,2,2,0  # this relates to how correlated variables are chosen

    #############
    # temperature
    idx_target = 2
    tdata=sio.loadmat('./results/NorthPole_var'+str(idx_target)+'_r'+str(polynomial_order)+'_tyrs'+str(years_train)+'_syrs'+str(start_year))
    y_truth,y_train,y_model=np.squeeze(tdata['y_truth']),np.squeeze(tdata['y_train']),tdata['ys_model']
    r_truth,p_truth,p_truth_u,p_truth_l=pt.pdf_1d_wCI(y_truth,nx=100,smoothing_sigma=F)
    r_train,p_train,p_train_u,p_train_l=pt.pdf_1d_wCI(y_train,nx=50,smoothing_sigma=F)

    for k in range(0,5):
        ax6 = myfig.add_axes([x0+k*(w+dw),y0-h-dh,w,h])
        ax6.tick_params(direction='in')
        plt.yticks([]),plt.xticks([])
        ax6.set_yscale('log')
        plt.yticks([1e-1,1e-3,1e-5])
        plt.ylim(1e-6,.8)
        plt.xlim(-10,10)
        plt.xticks([-5,5])

        if k ==0:
            plt.ylabel(r'$T_{NP}$',rotation=0)
            ax6.yaxis.set_label_coords(-.52,.5)
        else:
            ax6.set_yticklabels([])

        r_model,p_model=pt.pdf_1d(y_model[:,k],nx=100,smoothing_sigma=F,MyRange=[-20,20])
        plt.plot(r_model,p_model,color=MyColors[0],linewidth=lw0,label='model (74 years)')
        plt.plot(r_train,p_train,'k--',linewidth=lw1,label='train (2 years)')   
        plt.plot(r_truth,p_truth,'-.',linewidth=lw2,color=MyColors[1],label='truth (37 years)') 
        # confidence intervalo
        plt.fill_between(r_truth,p_truth_l,p_truth_u,alpha=0.35,color=MyColors[1],linewidth=0)
        plt.fill_between(r_train,p_train_l,p_train_u,alpha=0.35,color='k',linewidth=0)



        if k==2:
            legend=plt.legend(fontsize=tfs,bbox_to_anchor=(.5, -.43),loc='center',ncol=3,fancybox=True,framealpha=0)
            legend.get_frame().set_linewidth(0)
            legend.get_frame().set_edgecolor("black")
            



    # u-velocity
    idx_target = 0
    tdata=sio.loadmat('./results/NorthPole_var'+str(idx_target)+'_r'+str(polynomial_order)+'_tyrs'+str(years_train)+'_syrs'+str(start_year))
    y_truth,y_train,y_model=np.squeeze(tdata['y_truth']),np.squeeze(tdata['y_train']),tdata['ys_model']
    r_truth,p_truth,p_truth_u,p_truth_l=pt.pdf_1d_wCI(y_truth,nx=100,smoothing_sigma=F)
    r_train,p_train,p_train_u,p_train_l=pt.pdf_1d_wCI(y_train,nx=50,smoothing_sigma=F)


    for k in range(0,5):
        ax6 = myfig.add_axes([x0+k*(w+dw),y0,w,h])
        ax6.tick_params(direction='in')
        plt.yticks([])
        ax6.set_yscale('log')
        plt.yticks([1e-1,1e-3,1e-5])
        
        plt.ylim(1e-6,.8)
        plt.xlim(-12,12)
        plt.xticks([-10,10])

        if k ==0:
            plt.ylabel(r'$U_{NP}$',rotation=0)
            ax6.yaxis.set_label_coords(-.52,.5)
        else:
            ax6.set_yticklabels([])

        r_model,p_model=pt.pdf_1d(y_model[:,k],nx=100,smoothing_sigma=F,MyRange=[-20,20])
        plt.plot(r_model,p_model,color=MyColors[0],linewidth=lw0)
        plt.plot(r_train,p_train,'k--',linewidth=lw1)   
        plt.plot(r_truth,p_truth,'-.',linewidth=lw2,color=MyColors[1])
        

        # confidence intervalo - shaded enevelope method
        plt.fill_between(r_truth,p_truth_l,p_truth_u,alpha=0.35,color=MyColors[1],linewidth=0)
        plt.fill_between(r_train,p_train_l,p_train_u,alpha=0.35,color='k',linewidth=0)
        plt.title(r'$n='+str(k)+'$',fontsize=tfs)

    plt.savefig(savepath+'climate_tails.'+picformat,dpi=400)



if __name__=='__main__':
    """Runs the climate tail extrapolation and generates the plots in the paper."""
    print('extrapolating tails ...')
    tt = timeit.default_timer()
    ExtrapolateTails(idx_target=0)
    ExtrapolateTails(idx_target=2)
    Plot_NorthPole_signals('./')
    Plot_tails_wCI('./')



    print('whole computation took {} seconds'.format(timeit.default_timer() - tt))
