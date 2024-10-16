#%%
import numpy as np
from scipy.io import savemat
from Deep_Utils import MPS_single, Powerspectrum,dbscale,Correlation
import matplotlib.pyplot as plt
import os
#%%
""" Load tomograms"""
rootFolder = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
fnameTom = '//[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)' # fovea
tomShape = [(586,896,960,2)]# porcine cornea
name = 'Experimental'
fname = rootFolder + fnameTom
# Names of all real and imag .bin files
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]

tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using
# tomReal = np.sum(tomReal,axis=3)

tomImag = np.fromfile(fnameTomImag[0],'single')
tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using
# tomImag = np.sum(tomImag,axis=3)

tomDownReal = tomReal[:,:,1::2,:]
tomDownImag = tomImag[:,:,1::2,:]
#%%
tomsub = tomDownReal[:,:,:,1]+1j*tomDownImag[:,:,:,1]
# tomsub = np.sum(tomsub,axis=3)
tomsub = np.stack((tomsub.real,tomsub.imag),axis=3)
np.save(os.path.join(rootFolder,'[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)_subsampled.npy'),tomsub)
#%%
savepath = rootFolder + '\\tomDownReal.mat'
mdic = {"tomDownReal": tomDownReal, "label": "tomDownReal"}
savemat(savepath, mdic)

savepath = rootFolder + '\\tomDownImag.mat'
mdic = {"tomDownImag": tomDownReal, "label": "tomDownImag"}
savemat(savepath, mdic)