#%% 
import numpy as np
import os
import matplotlib.pyplot as plt
from Deep_Utils import dbscale
from matplotlib.colors import Normalize as cmnorm
from matplotlib.cm import ScalarMappable
import plotly.express as px
#%%
path = r'E:\DLOCT\TomogramsDataAcquisition\[DepthWrap]\[DepthWrap][NailBed][09-18-2023_10-54-07]'
OriginalFile = '[NailBed]z=1024_x=1152_y=512_pol=2'
imagFile = OriginalFile + '_imag.bin'
realFile = OriginalFile + '_real.bin'
z = 1024
x = 1152
y = 512
pol = 2
tomReal = np.fromfile(os.path.join(path,realFile),dtype='single')
tomReal = np.reshape(tomReal,(z,x,y,pol),order='F')
tomReal = np.sum(tomReal,axis=3)
tomImag = np.fromfile(os.path.join(path,imagFile),dtype='single')
tomImag = np.reshape(tomImag,(z,x,y,pol),order='F')
tomImag = np.sum(tomImag,axis=3)
tomOriginal = np.stack((tomReal,tomImag),axis=3)
del tomReal, tomImag
tomOriginal = np.flip(tomOriginal[300:800,:,0:256,:],axis=0)
#%%
fileT2pol1 = 'tomDataOver_z=1024_x=1152_y=256_pol1.npy'
fileT2pol2 = 'tomDataOver_z=1024_x=1152_y=256_pol2.npy'
tomsubsampledT2pol1 = np.load(os.path.join(path,fileT2pol1))
tomsubsampledT2pol2 = np.load(os.path.join(path,fileT2pol2))
tomSubsampledT2 = tomsubsampledT2pol1 + tomsubsampledT2pol2
del tomsubsampledT2pol1, tomsubsampledT2pol2
tomSubsampledT2 = np.flip(tomSubsampledT2[300:800,:,:],axis=0)
tomSubsampledT2 = np.stack((np.real(tomSubsampledT2),np.imag(tomSubsampledT2)),axis=3)
#%%
fileT3pol1 = 'tomDataOver_z=1024_x=1152_y=512_pol1.npy'
fileT3pol2 = 'tomDataOver_z=1024_x=1152_y=512_pol2.npy'
tomsubsampledT3pol1 = np.load(os.path.join(path,fileT3pol1))
tomsubsampledT3pol1 = tomsubsampledT3pol1[300:800,:,:]
tomsubsampledT3pol2 = np.load(os.path.join(path,fileT3pol2))
tomsubsampledT3pol2 = tomsubsampledT3pol2[300:800,:,:]
tomSubsampledT3 = tomsubsampledT3pol1 + tomsubsampledT3pol2
del tomsubsampledT3pol1, tomsubsampledT3pol2
tomSubsampledT3 = np.flip(tomSubsampledT3,axis=0)
tomSubsampledT3 = np.stack((np.real(tomSubsampledT3),np.imag(tomSubsampledT3)),axis=3)

#%%
vmin = 60
vmax = 120
n = 200
savefig = False
plotT1 = dbscale(tomOriginal[:,n,:,:])
plotT2 = dbscale(tomSubsampledT2[:,n,:,:])
plotT3 = dbscale(tomSubsampledT3[:,n,:])
fig, axs = plt.subplots(1, 3, figsize=(30, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(plotT1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
axs[0].set_title('T1')

axs[1].imshow(plotT2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
axs[1].set_title('T2')

axs[2].imshow(plotT3, cmap='gray',vmin=vmin,vmax=vmax, aspect = 'auto')
axs[2].axis('off')
axs[2].set_title('T3')
plt.subplots_adjust(wspace=0.01, hspace=0)
figname = f'comparision different T x={n}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB') 
folder = 'cortes'
subfolder = f'corte{6}'
pathimages = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
if savefig:
    plt.savefig(os.path.join(pathimages,folder,subfolder,figname), dpi=300)
    print('fig saved')
 
plt.show()
#%%
del tomSubsampledT2, tomSubsampledT3, tomOriginal
#%%
path = r'E:\DLOCT\TomogramsDataAcquisition\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'
OriginalFile = 'Tom_z=1152_x=1024_y=1024_pol=2'
imagFile = OriginalFile + '_imag.bin'
realFile = OriginalFile + '_real.bin'
z = 1152
x = 1024
y = 1024
pol = 2
tomReal = np.fromfile(os.path.join(path,realFile),dtype='single')
tomReal = np.reshape(tomReal,(z,x,y,pol),order='F')
tomReal = tomReal[400:960,:,0:512,:]
tomReal = np.sum(tomReal,axis=3)
tomImag = np.fromfile(os.path.join(path,imagFile),dtype='single')
tomImag = np.reshape(tomImag,(z,x,y,pol),order='F')
tomImag = tomImag[400:960,:,0:512,:]
tomImag = np.sum(tomImag,axis=3)
tomOriginal = np.stack((tomReal,tomImag),axis=3)
del tomReal, tomImag
#%%
fileT2pol1 = 'tomDataOver_z=560_x=1024_y=512_pol1.npy'
fileT2pol2 = 'tomDataOver_z=560_x=1024_y=512_pol2.npy'
tomsubsampledT2pol1 = np.load(os.path.join(path,fileT2pol1))
tomsubsampledT2pol2 = np.load(os.path.join(path,fileT2pol2))
tomSubsampledT2 = tomsubsampledT2pol1 + tomsubsampledT2pol2
del tomsubsampledT2pol1, tomsubsampledT2pol2
# tomSubsampledT2 = np.flip(tomSubsampledT2[300:800,:,:],axis=0)
tomSubsampledT2 = np.stack((np.real(tomSubsampledT2),np.imag(tomSubsampledT2)),axis=3)
#%%
fileT3pol1 = 'tomDataOver_z=560_x=1024_y=1024_pol1.npy'
fileT3pol2 = 'tomDataOver_z=560_x=1024_y=1024_pol2.npy'
tomsubsampledT3pol1 = np.load(os.path.join(path,fileT3pol1))
tomsubsampledT3pol1 = tomsubsampledT3pol1[:,0:100,:]
tomsubsampledT3pol2 = np.load(os.path.join(path,fileT3pol2))
tomsubsampledT3pol2 = tomsubsampledT3pol2[:,0:100,:]
tomSubsampledT3 = tomsubsampledT3pol1 + tomsubsampledT3pol2
del tomsubsampledT3pol1, tomsubsampledT3pol2
# tomSubsampledT3 = np.flip(tomSubsampledT3,axis=0)
tomSubsampledT3 = np.stack((np.real(tomSubsampledT3),np.imag(tomSubsampledT3)),axis=3)
#%%
vmin = 60
vmax = 120
n = 20
savefig = False
plotT1 = dbscale(tomOriginal[:,n,:,:])
plotT2 = dbscale(tomSubsampledT2[:,n,:,:])
plotT3 = dbscale(tomSubsampledT3[:,n,:])
fig, axs = plt.subplots(1, 3, figsize=(30, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(plotT1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
axs[0].set_title('T1')

axs[1].imshow(plotT2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
axs[1].set_title('T2')

axs[2].imshow(plotT3, cmap='gray',vmin=vmin,vmax=vmax, aspect = 'auto')
axs[2].axis('off')
axs[2].set_title('T3')
plt.subplots_adjust(wspace=0.01, hspace=0)
figname = f'comparision different T2 x={n}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB') 
folder = 'cortes'
subfolder = f'corte{6}'
pathimages = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
if savefig:
    plt.savefig(os.path.join(pathimages,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%
path = r'E:\DLOCT\TomogramsDataAcquisition\[DepthWrap]\[DepthWrap][ChickenBreast][09-18-2023_11-29-20]'
OriginalFile = 'Tom_z=1024_x=1024_y=512'
imagFile = OriginalFile + '_imag.bin'
realFile = OriginalFile + '_real.bin'
z = 1024
x = 1024
y = 512
tomReal = np.fromfile(os.path.join(path,realFile),dtype='single')
tomReal = np.reshape(tomReal,(z,x,y),order='F')
tomReal = tomReal[150:650,:,0:256]
tomImag = np.fromfile(os.path.join(path,imagFile),dtype='single')
tomImag = np.reshape(tomImag,(z,x,y),order='F')
tomImag = tomImag[150:650,:,0:256]
tomOriginal = np.stack((tomReal,tomImag),axis=3)
del tomReal, tomImag
#%%
fileT2pol1 = 'tomDataOver_z=1024_x=1024_y=256_pol1.npy'
tomSubsampledT2 = np.load(os.path.join(path,fileT2pol1))
tomSubsampledT2 = np.stack((np.real(tomSubsampledT2[150:650,:,:]),np.imag(tomSubsampledT2[150:650,:,:])),axis=3)
#%%
fileT3pol1 = 'tomDataOver_z=1024_x=1024_y=512_pol1.npy'
tomSubsampledT3= np.load(os.path.join(path,fileT3pol1))
tomSubsampledT3 = np.stack((np.real(tomSubsampledT3[150:650,:,:]),np.imag(tomSubsampledT3[150:650,:,:])),axis=3)
#%%
vmin = 60
vmax = 120
n = 460
savefig = True
plotT1 = dbscale(tomOriginal[:,n,:,:])
plotT2 = dbscale(tomSubsampledT2[:,n,:,:])
plotT3 = dbscale(tomSubsampledT3[:,n,:])
fig, axs = plt.subplots(1, 3, figsize=(30, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(plotT1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
axs[0].set_title('T1')

axs[1].imshow(plotT2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
axs[1].set_title('T2')

axs[2].imshow(plotT3, cmap='gray',vmin=vmin,vmax=vmax, aspect = 'auto')
axs[2].axis('off')
axs[2].set_title('T3')
plt.subplots_adjust(wspace=0.01, hspace=0)
figname = f'comparision different T3 x={n}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB') 
folder = 'cortes'
subfolder = f'corte{6}'
pathimages = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
if savefig:
    plt.savefig(os.path.join(pathimages,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()