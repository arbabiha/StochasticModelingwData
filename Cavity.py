"""
Stochastic modeling of SPOD coordiantes for lid-driven cavity flow.

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
from sklearn.neighbors import KNeighborsRegressor
import dill
import timeit
from joblib import Parallel, delayed
import multiprocessing
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# custom tools
from sys import path
path.append('./thehood/')
import plotting_tools as pt
import spectral_analysis as sa
import transport_maps as tm
import SDE_systems as sm


def Cavity_SDEmodeling():
    """Models a state variable of Lorenz using optimal transport.
    
    The code
    1- loads the SPOD mode and coordiante data,
    2- computes the transport map to standard normal distiburion,
    3- identifies linear stochastic oscillators with same spectra
    4- computes a trajectory from oscillators and pull them back under the transport map.
    5- computes pointwise stats for cavity
    """

    Dim = 10
    n_samp,t_max=10,2500
    poly_order=2

    # data tag
    Tag='production_T'+str(t_max)+'/'
    print('data tag= '+Tag)
    print('Dim='+str(Dim)+' n_samp='+str(n_samp)+' t_max='+str(t_max) )

    ## save path
    SavePath='./results/'
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    # We have already computed the SPOD of cavity
    # here we only load and subsample the data

    CavityData=sio.loadmat('./thehood/Cavity_SPOD_small.mat') 
    SPOD_coords,t_coords = CavityData['y'],CavityData['t']
    SPOD_coords = np.real(SPOD_coords[0:Dim,:])
    SPOD_coords = np.swapaxes(SPOD_coords,0,1)
    n_max = int(t_max/(t_coords[1]-t_coords[0]))



    y_train = SPOD_coords[:n_max:n_samp,:] # training data
    t_train = t_coords[:n_max:n_samp]
    dt = t_train[1]-t_train[0]
    print('train data size='+str(y_train.shape))

    # form the map and transform data
    T=tm.compute_transport_map(y_train,polynomial_order=poly_order,MPIsetup=None)
    q_train = T(y_train)   # training data for q


    # # save the transport map
    # file_h = open(SavePath+'CTM.dll', 'wb') 
    # dill.dump(T,file_h)
    # file_h.close()

    
    Oscillator_Params=sm.SystemID_spec_match(q_train,dt=dt) # model the q dynamics


    # generate trajectory of those SDEs
    t_model,q_model=sm.draw_trajectory_SDEsys(Oscillator_Params,T=10000,dt=dt)

    # # model data
    # np.savez(SavePath+'Cavity_Oscillator_Params',Params=Oscillator_Params)


    # compute inverse 
    print('computing the exact inverse ...')
    y_model=tm.TMinverse(q_model,T,num_core = 20)


    # save model data for MATLAB
    sio.savemat(SavePath+'Cavity_modal_'+Tag[:-1],{'y_model':y_model,'y_train':y_train,'t_model':t_model,'t_train':t_train,'q_model':q_model,'q_train':q_train})


    # compute the poitwsie stat in truth and model
    PointwiseStats4cavity(y_train,y_model,SPOD_coords,Tag) 

    return Tag[:-1]


def PointwiseStats4cavity(x_train,x_model,x_truth,Tag):
    """Computes the pointwise time-series at several points in cavity.

    This code constructs the SPOD model -- using SDE model and truth --
    and computes time-series of pointwise velocity at a few points.
    Loads the sensor location information.

    Args:
        x_train (np.ndarray): the training SPOD coordiante time series
        x_model (np.ndarray): the SPOD coordiante time series from the stochastic model
        x_truth (np.ndarray): the truth SPOD coordiante time series

    Returns:
        saves the ponitwise time series.
    """

    # load spatial modes 
    Dim = x_model.shape[1]
    ModeData=sio.loadmat('./thehood/Cavity_SPOD_small.mat')
    Modes,Gram = ModeData['ModeMatrix'],ModeData['ModeGram']
    Modes=np.real(Modes[:,0:Dim])
    Gram=Gram[0:Dim,0:Dim]

    # sensor locations
    SensorData=sio.loadmat('./thehood/CavitySensors.mat')
    Sensors = np.squeeze(SensorData['SensorIndex']) - 1
    Sensors = np.concatenate((Sensors,40000+Sensors),axis=0)

    # modes are not orthogonal so ...
    c_truth=np.real(np.linalg.solve(Gram,np.swapaxes(x_truth,0,1)))
    c_train=np.real(np.linalg.solve(Gram,np.swapaxes(x_train,0,1)))
    c_model=np.real(np.linalg.solve(Gram,np.swapaxes(x_model,0,1)))

    # # iterate over data and collect the points
    n_truth = min(c_truth.shape[1],c_model.shape[1])
    n_train= c_train.shape[1]

    uv_truth =np.zeros((Sensors.shape[0],n_truth))
    uv_model =np.zeros((Sensors.shape[0],n_truth))
    uv_train =np.zeros((Sensors.shape[0],n_train))


    n_chunk = 1000

    print('iterating the flow to compute pointwise stat')
    t_sim = timeit.default_timer()

    for k in range(int(n_truth/n_chunk)):
        
        k1,k2 = k*n_chunk,(k+1)*n_chunk
        
        Flow_truth = np.matmul(Modes,c_truth[:,k1:k2])
        uv_truth[:,k1:k2]=Flow_truth[Sensors,:]
        
        Flow_model = np.matmul(Modes,c_model[:,k1:k2])
        uv_model[:,k1:k2]=Flow_model[Sensors,:]    
        
        if k % 10 ==9:
            print(str(k+1)+' / '+str(n_truth/n_chunk)+' done')

    n_chunk = 50      
    for k in range(int(n_train/n_chunk)):
        
        k1,k2 = k*n_chunk,(k+1)*n_chunk
        
        Flow_train = np.matmul(Modes,c_train[:,k1:k2])
        uv_train[:,k1:k2]=Flow_train[Sensors,:]
        
        if k % 10 ==9:
            print(str(k+1)+' / '+str(n_truth/n_chunk)+' done')
        
        
    print('Flow iteration took {} seconds'.format(timeit.default_timer() - t_sim))

    # # save pointwise data
    sio.savemat('./results/Cavity_pointwise_'+Tag[:-1],{'uv_model':uv_model,'uv_train':uv_train,'uv_truth':uv_truth,'Sensors':Sensors})


def Plot_SPOD_marginal(savepath,tag='',picformat='png'):
    """Plots the PDF marginals of SPOD coordiantes."""

    plt.rc('font', family='serif',size=9)
    tfs = 10

    TruthData=sio.loadmat('./thehood/Cavity_SPOD_small.mat')
    y_truth=np.real(TruthData['y'])
    y_truth= np.swapaxes(y_truth[:10,:],0,1)
    ModalData=sio.loadmat('./results/Cavity_modal_'+tag+'.mat')
    y_model,y_train,t=ModalData['y_model'],ModalData['y_train'],ModalData['t_model']

    mycolors='Blues'
    lims=[-.026,0.026]
    

    # comparison of marginals for 4 variables
    vindex=[1,3,6,9]
    mytitles=[r'$y_2$',r'$y_4$',r'$y_6$',r'$y_{10}$']
    pt.Marginals_plot(y_truth[:,vindex],nx=100,ny=100,Titles=mytitles,figsize=[3.,3.],Colors=mycolors,ticks=[],lims=lims,tfs=tfs)
    plt.savefig(savepath+'cavity_marginal_truth_4d'+tag+'.'+picformat,dpi=350)

    pt.Marginals_plot(y_model[:,vindex],nx=100,ny=100,Titles=mytitles,figsize=[3.,3.],Colors=mycolors,ticks=[],lims=lims,tfs=tfs)
    plt.savefig(savepath+'cavity_marginal_model_4d'+tag+'.'+picformat,dpi=350)

    # # comparison of marginals for all 10 variables
    vindex=range(10)
    mytitles=[r'$y_1$',r'$y_2$',r'$y_3$',r'$y_4$',r'$y_5$',r'$y_6$',r'$y_7$',r'$y_8$',r'$y_9$',r'$y_{10}$']
    pt.Marginals_plot(y_truth[:,vindex],nx=100,ny=100,Titles=mytitles,figsize=[8,8],Colors=mycolors,ticks=[],lims=lims,tfs=tfs)
    plt.savefig(savepath+'cavity_marginal_truth_10d'+tag+'.'+picformat,dpi=300)

    pt.Marginals_plot(y_model[:,vindex],nx=100,ny=100,Titles=mytitles,figsize=[8,8],Colors=mycolors,ticks=[],lims=lims,tfs=tfs)
    plt.savefig(savepath+'cavity_marginal_model_10d'+tag+'.'+picformat,dpi=300)


def Plot_signals_and_spectra(savepath,tag='',picformat='png'):
    """Plots the pointwise SPOD signals and spectra."""

    plt.rc('font', family='serif',size=9)
    tfs = 10
    MyColors=['#377eb8','#d95f02']


    ModalData=sio.loadmat('./results/Cavity_modal_'+tag+'.mat')
    y_model,y_train,t_model,t_train=ModalData['y_model'],ModalData['y_train'],np.squeeze(ModalData['t_model']),np.squeeze(ModalData['t_train'])
    t_model= t_model[0:y_model.shape[0]]



    plt.figure(figsize=[6.5,6])

    vindex=range(10)
    ylimits=[-.025,.025]
    xlimits=[0,50]
    ncols=2

    for j in vindex:
        ym = y_model[:,j]
        yt = y_train[:,j]

        plt.subplot(10,ncols,j*ncols+1)
        plt.plot(t_train-100,yt,color=MyColors[0])
        plt.xlim(xlimits),plt.xticks([])
        plt.ylim(ylimits),plt.yticks([-.02,.02])
        plt.tick_params(direction='in')


        if j==9:
            plt.xticks([0,10,20,30,40,50])
            plt.xlabel(r'$t$',fontsize=tfs)
        if j==0:
            plt.title(r'truth signal',fontsize=tfs)
        plt.ylabel(r'$y_{'+str(j+1)+'}$',fontsize=tfs,rotation=0)

        plt.subplot(10,ncols,j*ncols+2)
        plt.plot(t_model-100,ym,color=MyColors[0])
        plt.xlim(xlimits),plt.xticks([])
        plt.ylim(ylimits),plt.yticks([])
        
        if j==9:
            plt.xticks([0,10,20,30,40,50])
            plt.xlabel(r'$t$',fontsize=tfs)
        if j==0:
            plt.title(r'SDE model signal',fontsize=tfs)



    plt.savefig(savepath+'cavity_signals_10d.'+picformat,dpi=400)



    q_model,q_train=ModalData['q_model'],ModalData['q_train']

    plt.figure(figsize=[6.5,6])
    vindex=range(10)
    ylimits=[-.025,.025]
    xlimits=[0,50]
    ncols=2

    for j in vindex:
        qm = q_model[:,j]
        qt = q_train[:,j]

        plt.subplot(10,ncols,j*ncols+2)
        fsm = 1/t_model[1]-t_model[0]
        print(fsm)
        rm,pm= sa.Welch_estimator(qm,M=512,L=512,fs=fsm)
        plt.plot(rm,pm,'k',label=r'SDE model')
        plt.xlim(0,np.pi*fsm/5)

        fst = 1/t_train[1]-t_train[0]
        rr,pr= sa.Welch_estimator(qt,M=512,L=512,fs=fst)
        plt.plot(rr,pr,'--',color='gray',label=r'truth')
        plt.xticks([])
        plt.yticks([0,int(np.amax(pr))])

        if j==9:
            plt.xticks([0,2,4,6])
            plt.xlabel(r'$\omega$',fontsize=tfs)
        if j==0:
            plt.title(r'PSD of $q_j=T_j(y)$',fontsize=tfs)


        yt = y_train[:,j]
        ym = y_model[:,j]
        ym[np.abs(ym)>.05]=0  


        ax=plt.subplot(10,ncols,j*ncols+1)
        rm,pm= sa.Welch_estimator(ym,M=512,L=512,fs=fsm)
        plt.plot(rm,pm,'k',label=r'SDE model')
        plt.xlim(0,np.pi*fsm/5)

        rr,pr= sa.Welch_estimator(yt,M=512,L=512,fs=fst)
        plt.plot(rr,pr,'--',color='gray',label=r'truth')
        plt.xticks([])
        # plt.ylabel(r'$j='+str(j+1)+'$',fontsize=tfs-1,rotation=90)
        plt.text(-.33,.5,r'$j='+str(j+1)+'$',fontsize=tfs-1,transform=ax.transAxes)
        plt.yticks([0,int(np.amax(pr*1e4))/1e4])
        ax.set_yticklabels(['0',str(int(np.amax(pr*1e4)))+'e-4'])
        
        if j==9:
            plt.xticks([0,2,4,6])
            plt.xlabel(r'$\omega$',fontsize=tfs)
        if j==0:
            plt.title(r'PSD of $y_j$',fontsize=tfs)
            legend=plt.legend(fontsize=tfs-1,bbox_to_anchor=(1.03, .55),loc='center right',ncol=1,fancybox=True,framealpha=0)
            legend.get_frame().set_linewidth(0)
        


    plt.savefig(savepath+'cavity_spectra_10d.'+picformat,dpi=400)

def Plot_pointwise_stat(savepath,tag='',picformat='png'):
    """Plots the pointwise SPOD signals and spectra."""

    plt.rc('font', family='serif',size=9)
    tfs = 10



    PointWiseData=sio.loadmat('./results/Cavity_pointwise_'+tag+'.mat')
    CavityField = sio.loadmat('./thehood/CavitySensors.mat')
    



    # what sensors to look at
    s= [12,22,23,47]  # out of 0-48

    SensorIndex = np.squeeze(CavityField['SensorIndex'])
    ns = SensorIndex.shape[0]
    # SensorIndex=SensorIndex[s] # -1 is for python vs MATLAB indexing

    # print(SensorIndex)

    uv_model=PointWiseData['uv_model']
    uv_truth=PointWiseData['uv_truth']
    print(uv_model.shape)
    print(uv_truth.shape)


    # funcion that designates the position of axes
    apr= 1/2  # aspect ratio
    myfig=plt.figure(figsize=[6.5,6.5*apr])
    def CreateAxes(i,j,myfig):
        # i,j are the row column indices of axes from top left
        # xleft, ybottom
        h,w,dw,dh=.24,.24*apr,.03,.15
        x0,y0 = 0.34,.6

        xl = x0 + (j-1)*(w+dw)
        yl = y0 - (i-1)*(h+dh)

        my_axes = myfig.add_axes([xl,yl,w,h])
        plt.tick_params(direction='in')
        plt.yticks([])
        # plt.xticks([])
        return my_axes  


    xtiko = np.array([[-1,0,1],[-.2,0,.2],[-1,0,1],[-.6,0,.6],
                        [-1,0,1],[-.25,0,.25],[-.5,0,.5],[-2,0,2]])



    for j in range(4):
        ax2=CreateAxes(1,j+1,myfig)
        # u-velocity
        y = uv_truth[s[j],:]
        x0,p0=pt.pdf_1d(y,nx=100,smoothing_sigma=3)
        xl=[np.amin(x0),np.amax(x0)]


        # model
        y = uv_model[s[j],:]
        x,p=pt.pdf_1d(y,nx=100,smoothing_sigma=3,MyRange=1.5*np.array(xl))
        plt.plot(x,p,'k',label='SDE model PDF')
        plt.plot(x0,p0,'k-.',color='gray',label='true PDF')
        plt.xticks(xtiko[j,:])
        plt.title(r'$u_'+str(j+1)+'$',fontsize=tfs)
        
        
        # ax2.set_yscale('log')
        if j==0:
            legend=plt.legend(fontsize=tfs,bbox_to_anchor=(4.5, -1.9),ncol=2,fancybox=True)
            legend.get_frame().set_linewidth(0)
            legend.get_frame().set_edgecolor("black")
        ax2=CreateAxes(2,j+1,myfig)
        # v-velocity
        y = uv_truth[s[j]+ns,:]
        x0,p0=pt.pdf_1d(y,nx=100,smoothing_sigma=3)
        xl=[np.amin(x0),np.amax(x0)]


        # model
        y = uv_model[s[j]+ns,:]
        x,p=pt.pdf_1d(y,nx=100,smoothing_sigma=3,MyRange=1.5*np.array(xl))
        plt.plot(x,p,'k')
        plt.plot(x0,p0,'k-.',color='gray') 
        plt.xticks(xtiko[j+4,:])
        plt.title(r'$v_'+str(j+1)+'$',fontsize=tfs)


    # cavity snapshot
    x,y,vort,cm=CavityField['x'],CavityField['y'],CavityField['vort'],CavityField['colormap_vort']
    X,Y=np.squeeze(CavityField['X']),np.squeeze(CavityField['Y'])
    cm = np.concatenate((cm,np.ones((cm.shape[0],1))),axis=1)
    cm = ListedColormap(cm)


    ax1 = myfig.add_axes([.05,.28,.45*apr,.45])
    plt.contourf(x,y,vort,100)
    plt.set_cmap(cm)
    plt.xticks([]),plt.yticks([])

    xpa,ypa=[.07,.08,-.08,.1],[.06,.06,-.24,-.14]


    for j in range(4):
        xp,yp=X[SensorIndex[s[j]]-1],Y[SensorIndex[s[j]]-1]
        ax1.plot(xp,yp,'kx')
        ax1.text(xp+xpa[j],yp+ypa[j],r''+str(j+1),fontsize=tfs-1)


    plt.savefig(savepath+'cavity_pointwise.'+picformat,dpi=400)



if __name__ == '__main__':
    """Runs the cavity modeling and generates the plots in the paper."""
    print('modeling cavity flow  ...')
    tt = timeit.default_timer()
    tag=Cavity_SDEmodeling()
    Plot_SPOD_marginal('./',tag=tag)
    Plot_signals_and_spectra('./',tag=tag)
    Plot_pointwise_stat('./',tag=tag)
    print('whole computation took {} seconds'.format(timeit.default_timer() - tt))
    