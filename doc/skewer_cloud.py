'''

skewer versus cloud parameter space sampling test

skewer: 
    |       x
    |   x   x
  X |   x   x
    |   x
    |-----------
        theta 
cloud: 
    |        x
    |   x x x
  X |  xx  x
    |   
    |-----------
        theta 
'''
import os 
import h5py 
import numpy as np 
# --- 
from sklearn.gaussian_process import GaussianProcessRegressor as GPR 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
# --- plotting --- 
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def gpPeakcounts(): 
    ''' read in peak counts and train GP 
    '''
    # read in peak counts data
    datdir = os.path.join(os.environ['MNULFI_DIR'], 'Peaks_MassiveNuS')
    thetas = np.load(os.path.join(datdir, 'params_conc_means.npy')) # thetas 
    peakct = np.load(os.path.join(datdir, 'data_scaled_means.npy')) # average peak counts 

    kern = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4)) # kernel
    gp = GPR(kernel=kern, n_restarts_optimizer=10) # instanciate a GP model
    gp.fit(thetas, peakct)
    return gp 


def makeSkewerCloud(seed=1): 
    ''' Generate skewer and cloud data using Gaussian Process
    '''
    np.random.seed(seed)
    # read in covariance 
    datdir = os.path.join(os.environ['MNULFI_DIR'], 'Peaks_MassiveNuS')

    thetas = np.load(os.path.join(datdir, 'params_conc_means.npy')) # thetas 
    cov = np.load(os.path.join(datdir, 'covariance.npy')) # covariance  

    GP = gpPeakcounts() # gaussian process 
    
    # generate skewer data 
    theta_skewers = np.tile(thetas[None,:,:], 9999).reshape(thetas.shape[0] * 9999, thetas.shape[1])
    peaks_skewers = np.random.multivariate_normal(np.zeros(50), cov, size=theta_skewers.shape[0])
    for iskew in range(thetas.shape[0]): 
        mu_peak = GP.predict(np.atleast_2d(thetas[iskew,:]))
        peaks_skewers[iskew*9999:(iskew+1)*9999,:] += mu_peak 

    theta_lims = [(theta_skewers[:,i].min(), theta_skewers[:,i].max()) for i in range(theta_skewers.shape[1])]

    # generate cloud data 
    theta_cloud = np.zeros_like(theta_skewers)
    theta_cloud[:,0] = np.random.uniform(theta_lims[0][0], theta_lims[0][1], theta_skewers.shape[0])
    theta_cloud[:,1] = np.random.uniform(theta_lims[1][0], theta_lims[1][1], theta_skewers.shape[0])
    theta_cloud[:,2] = np.random.uniform(theta_lims[2][0], theta_lims[2][1], theta_skewers.shape[0])

    peaks_cloud = np.random.multivariate_normal(np.zeros(50), cov, size=theta_cloud.shape[0])
    mu_peaks = GP.predict(theta_cloud)
    peaks_cloud += mu_peak
    
    # plot thetas 
    fig = plt.figure(figsize=(8,4))  
    sub = fig.add_subplot(121) 
    sub.scatter(theta_cloud[::100,0], theta_cloud[::100,1], c='C0', label='Cloud') 
    sub.scatter(theta_skewers[::100,0], theta_skewers[::100,1], c='C1', label='Skewer') 
    sub.set_xlabel(r'$\theta_1$', fontsize=20) 
    sub.set_xlim(theta_lims[0])
    sub.set_ylabel(r'$\theta_2$', fontsize=20) 
    sub.set_ylim(theta_lims[1])

    sub = fig.add_subplot(122) 
    sub.scatter(theta_cloud[::100,2], theta_cloud[::100,1], c='C0', label='Cloud') 
    sub.scatter(theta_skewers[::100,2], theta_skewers[::100,1], c='C1', label='Skewer') 
    sub.legend(loc='lower right', frameon=True, fontsize=15) 
    sub.set_xlabel(r'$\theta_3$', fontsize=20) 
    sub.set_xlim(theta_lims[2])
    sub.set_ylim(theta_lims[1])
    fig.savefig(os.path.join(datdir, 'theta.skewer_cloud.png'), bbox_inches='tight') 

    # plot peak counts 
    fig = plt.figure(figsize=(8,8))  
    sub = fig.add_subplot(211) 
    for i in range(1000): 
        sub.plot(range(50), peaks_cloud[1000*i,:], c='k') 
    sub.set_xlim(0, 49)

    sub = fig.add_subplot(212) 
    for i in range(1000): 
        sub.plot(range(50), peaks_skewers[1000*i,:], c='k') 
    sub.set_xlim(0, 49)
    bkgd = fig.add_subplot(111, frameon=False) # x,y labels
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_ylabel('peak counts', fontsize=20) 
    fig.savefig(os.path.join(datdir, 'peaks.skewer_cloud.png'), bbox_inches='tight') 

    # save skewer and cloud to files
    theta_cloud.dump(os.path.join(datdir, 'theta.cloud.npy')) 
    peaks_cloud.dump(os.path.join(datdir, 'peaks.cloud.npy')) 

    theta_skewers.dump(os.path.join(datdir, 'theta.skewers.npy')) 
    peaks_skewers.dump(os.path.join(datdir, 'peaks.skewers.npy')) 
    return None


if __name__=="__main__":
    makeSkewerCloud()
