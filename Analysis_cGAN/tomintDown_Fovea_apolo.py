#%%
import numpy as np
from scipy.io import savemat

#%%
""" Load tomograms"""
rootFolder = '/home/dapulgaris/Data/' # apolo
#rootFolder = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected' # own pc
fnameTom = '//[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)' # fovea
tomShape = [(586,896,960,2,2)]# porcine cornea
name = 'Experimental'
fname = rootFolder + fnameTom
# Names of all real and imag .bin files
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]

tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using

tomImag = np.fromfile(fnameTomImag[0],'single')
tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using

tomData = np.stack((tomReal,tomImag), axis=5)
tomData = np.sum(tomData,axis=3) # Z,X,Y,pol1-2,imag-real

tomInt = abs(tomData[:, :, :,:, 0] + 1j*tomData[:, :, :,:,1])**2
tomIntDonwcomp = tomInt[:,1::2,:,:]
savepath = rootFolder + 'tomDataIntDown.mat'
mdic = {"tomintDown": tomIntDonwcomp, "label": "tomintDown"}
savemat(savepath, mdic)