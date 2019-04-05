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
# --- plotting --- 
import matplotlib as mpl
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


def readPeakcounts(): 
    ''' read in MassiveNuS peak counts data 
    '''
    fmnu = os.path.join(os.environ['MNULFI_DIR'], 
            'Peaks_MassiveNuS', 'MassiveNuS.hdf5')
    massnu = h5py.File(fmnu, 'r') 
    thetas = massnu['theta'].value      # thetas 
    peakct = massnu['peakcounts'].value # peak counts 

    return 


def makeSkewer(): 
    pass


def makeCloud(): 
    pass 


if __name__=="__main__":
