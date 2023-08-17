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
#%% predicted by network
pathcGAN = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected\cGAN_dr'
filename = 'tomDataOverpol'
mat_contents = sio.loadmat(pathcGAN+'\\'+filename+'0')
tomDataOver0 = mat_contents['tomDataOver']
filename = 'tomDataOverpol'
mat_contents = sio.loadmat(pathcGAN+'\\'+filename+'1')
tomDataOver1 = mat_contents['tomDataOver']
tomDataover = np.stack((tomDataOver0,tomDataOver1),axis=3)
tomDataover = np.sum(tomDataover,axis=3)
enface_over = tomDataover[z,:,:]
#%%
del tomDataover, tomDataOver0, tomDataOver1
#%%
plot_orig = 10*np.log10(abs(enface_original)**2)
plot_over = 10*np.log10(abs(enface_over)**2)
create_and_save_subplot(plot_orig,plot_over,
                        'Original','Reconstructed',
                        pathcGAN,zmin=30,zmax=150)

#%%
mps_orig = MPS_single(enface_original,meandim=0)
mps_reconstructed = MPS_single(enface_over,meandim=0)

