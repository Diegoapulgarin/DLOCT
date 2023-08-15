#%%
import sys
sys.path.append(r'C:\Data\DLOCT\cGAN_subsampling\Functions')
import numpy as np 
from Deep_Utils import create_and_save_subplot,Powerspectrum,MPS_single
import scipy.io as sio
#%%
pathorig = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
tomreal = np.fromfile(pathorig+'\\'+filename+real,'single')
tomreal = np.reshape(tomreal,(586,896,960,2),order='F')
tomreal = np.sum(tomreal,axis=3)
tomimag = np.fromfile(pathorig+'\\'+filename+imag,'single')
tomimag = np.reshape(tomimag,(586,896,960,2),order='F')
tomimag = np.sum(tomimag,axis=3)
z = 128
enface_original = tomreal[z,:,:]+1j*tomimag[z,:,:]
del tomimag, tomreal
#%%
pathcGAN = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected\cGAN1'
filename = 'tomDataOverpol'
mat_contents = sio.loadmat(pathcGAN+'\\'+filename+'0')
tomDataOver0 = mat_contents['tomDataOver']
filename = 'tomDataOverpol'
mat_contents = sio.loadmat(pathcGAN+'\\'+filename+'1')
tomDataOver1 = mat_contents['tomDataOver']
tomDataover = np.zeros((586,896,960,2))
tomDataover[:,:,:,0] = tomDataOver0
tomDataover[:,:,:,1] = tomDataOver1
tomDataover
#%%

#%%

mat_contents = sio.loadmat(path+'/'+filename)
fringes1 = mat_contents['fringes1']