#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
from Deep_Utils import Correlation, MPS_single, Powerspectrum, calculate_mse,sharpness,non_local_means_despeckling_3d,dbscale
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from matplotlib.colors import Normalize as cmnorm
from matplotlib.cm import ScalarMappable
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm
import cv2
import pandas as pd
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
#%%
path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 586
nXbin = 896
nYbin = 960
npol = 2
tom = np.fromfile(path+'\\'+filename+real,'single')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,npol),order='F')
tom = np.sum(tom,axis=3)
tomi = np.fromfile(path+'\\'+filename+imag,'single')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,npol),order='F')
tomi = np.sum(tomi,axis=3)
tomOriginal = np.stack((tom, tomi), axis=3)
del tom, tomi
print('original loaded')
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed'
tom = np.fromfile(path+'\\'+filename+real,'single')
tom = np.reshape(tom,(nZbin,nXbin,nYbin),order='F')
tomi = np.fromfile(path+'\\'+filename+imag,'single')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin),order='F')
tomReconstructed = np.stack((tom, tomi), axis=3)
del tom, tomi
print('reconstructed loaded')
tomSubsampled = tomOriginal[:,:,1::2,:]

# filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..480)_subsampled'
# nYbins = int(nYbin/2)
# tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(nZbin,nXbin,nYbins,npol),order='F')
# tom = np.sum(tom,axis=3)
# tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(nZbin,nXbin,nYbins,npol),order='F')
# tomi = np.sum(tomi,axis=3)
# tomSubsampled = np.stack((tom, tomi), axis=3)
# del tom, tomi

print('subsampled loaded')

# tomSubsampledInterp = np.zeros((nZbin,nXbin,nYbin,2))
# for z in tqdm(range(np.shape(tomSubsampled)[0])):
#     tomSubsampledInterp[z,:,:,:] = cv2.resize(tomSubsampled[z,:,:,:],
#                 dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
#                 interpolation=cv2.INTER_LINEAR)
# print('tom linearlly interpolated')

# tomSubsampledInterpBi = np.zeros((nZbin, nXbin, nYbin, 2))
# for z in tqdm(range(np.shape(tomSubsampled)[0])):
#     tomSubsampledInterpBi[z, :, :, :] = cv2.resize(
#         tomSubsampled[z, :, :, :], 
#         dsize=(int(np.shape(tomSubsampled)[2]*2), np.shape(tomSubsampled)[1]),
#         interpolation=cv2.INTER_CUBIC)
# print('tom cubic interpolated')


scale_factors = [1, 1, 2, 1]  # No escalas en z, escalas en x, no escalas en el canal

# Interpolación nearest
tomSubsampledInterp = zoom(tomSubsampled, zoom=scale_factors, order=0)
print('tom interpolated with nearest neighbor using scipy.ndimage.zoom')

# Interpolación bilineal
tomSubsampledInterpBi = zoom(tomSubsampled, zoom=scale_factors, order=1)
print('tom interpolated with bilinear using scipy.ndimage.zoom')

# # Interpolación cúbica
# tomSubsampledInterpCubic = zoom(tomSubsampled, zoom=scale_factors, order=3)
# print('tom interpolated with cubic using scipy.ndimage.zoom')


#%%
z = 170
x = 519
folder = 'cortes'
subfolder = f'corte{6}'
savefig = False
savefigindividuals = False
vmin= 75
vmax = 120
vmin2 = 70

plot1x = dbscale(tomReconstructed[:,x,:,:])
plot2x = dbscale(tomOriginal[:,x,:,:])
plot3x = dbscale(tomSubsampledInterp[:,x,:,:])
plot4x = dbscale(tomSubsampledInterpBi[:,x,:,:])

fig, axs = plt.subplots(1, 4, figsize=(30, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(plot2x, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
axs[0].set_title('Original')

axs[1].imshow(plot1x, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
axs[1].set_title('cGAN reconstructed')

axs[2].imshow(plot3x, cmap='gray',vmin=vmin2,vmax=vmax, aspect = 'auto')
axs[2].axis('off')
axs[2].set_title('Linear Interpolation')

axs[3].imshow(plot4x, cmap='gray',vmin=vmin2,vmax=vmax,aspect='auto')
axs[3].axis('off')
axs[3].set_title('Cubic Interpolation')

plt.subplots_adjust(wspace=0.01, hspace=0)
figname = f'comparision x={x}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB') 
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

#%%

plot1 = dbscale(tomReconstructed[z,:,:,:])
plot2 = dbscale(tomOriginal[z,:,:,:])
plot3 = dbscale(tomSubsampledInterp[z,:,:,:])
plot4 = dbscale(tomSubsampledInterpBi[z,:,:,:])

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(plot2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
axs[0].set_title('Original')

axs[1].imshow(plot1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
axs[1].set_title('cGAN reconstructed')

axs[2].imshow(plot3, cmap='gray',vmin=vmin2,vmax=vmax, aspect = 'auto')
axs[2].axis('off')
axs[2].set_title('Linear Interpolation')

axs[3].imshow(plot4, cmap='gray',vmin=vmin2,vmax=vmax,aspect='auto')
axs[3].axis('off')
axs[3].set_title('Cubic Interpolation')

plt.subplots_adjust(wspace=0.01, hspace=0)
figname = f'comparision z={z}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB') 
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%

aFiltered = non_local_means_despeckling_3d(plot1,search_window_size=21, block_size=5)
bFiltered = non_local_means_despeckling_3d(plot2,search_window_size=21, block_size=5)
cFiltered = non_local_means_despeckling_3d(plot3,search_window_size=21, block_size=5)
dFiltered = non_local_means_despeckling_3d(plot4,search_window_size=21, block_size=5)

axFiltered = non_local_means_despeckling_3d(plot1x,search_window_size=21, block_size=5)
bxFiltered = non_local_means_despeckling_3d(plot2x,search_window_size=21, block_size=5)
cxFiltered = non_local_means_despeckling_3d(plot3x,search_window_size=21, block_size=5)
dxFiltered = non_local_means_despeckling_3d(plot4x,search_window_size=21, block_size=5)
#%%
xint = 500
xfin = 630
yint = 157
yfin = 340
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot1 = plot1[xint:xfin,yint:yfin]
miniplot2 = plot2[xint:xfin,yint:yfin]
# miniplot3 = plot3[xintsub:xfinsub,yintsub:yfinsub]
miniplot3 = plot3[xint:xfin,yint:yfin]
miniplot4 = plot4[xint:xfin,yint:yfin]
ash = np.round(sharpness(miniplot1),decimals=2)
bsh = np.round(sharpness(miniplot2),decimals=2)
csh = np.round(sharpness(miniplot3),decimals=2)
dsh = np.round(sharpness(miniplot4),decimals=2)
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[1].imshow(miniplot1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[2].imshow(miniplot3, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot4, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%

xint = 83
xfin = 183
yint = 300
yfin = 550
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot1 = plot1[xint:xfin,yint:yfin]
miniplot2 = plot2[xint:xfin,yint:yfin]
# miniplot3 = plot3[xintsub:xfinsub,yintsub:yfinsub]
miniplot3 = plot3[xint:xfin,yint:yfin]
miniplot4 = plot4[xint:xfin,yint:yfin]
ash = np.round(sharpness(miniplot1),decimals=2)
bsh = np.round(sharpness(miniplot2),decimals=2)
csh = np.round(sharpness(miniplot3),decimals=2)
dsh = np.round(sharpness(miniplot4),decimals=2)
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[1].imshow(miniplot1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[2].imshow(miniplot3, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot4, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%
x2 = 135
plot1x = dbscale(tomReconstructed[:,x2,:,:])
plot2x = dbscale(tomOriginal[:,x2,:,:])
plot3x = dbscale(tomSubsampledInterp[:,x2,:,:])
plot4x = dbscale(tomSubsampledInterpBi[:,x2,:,:])
xint = 120
xfin = 230
yint = 300
yfin = 550
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot1 = plot1x[xint:xfin,yint:yfin]
miniplot2 = plot2x[xint:xfin,yint:yfin]
# miniplot3 = plot3x[xintsub:xfinsub,yintsub:yfinsub]
miniplot3 = plot3x[xint:xfin,yint:yfin]
miniplot4 = plot4x[xint:xfin,yint:yfin]
ash = np.round(sharpness(miniplot1),decimals=2)
bsh = np.round(sharpness(miniplot2),decimals=2)
csh = np.round(sharpness(miniplot3),decimals=2)
dsh = np.round(sharpness(miniplot4),decimals=2)
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[1].imshow(miniplot1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[2].imshow(miniplot3, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot4, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered x={x2}_zint{xint}_zfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%

y = 423
plot1y = dbscale(tomReconstructed[:,:,y,:])
plot2y = dbscale(tomOriginal[:,:,y,:])
plot3y = dbscale(tomSubsampledInterp[:,:,y,:])
plot4y = dbscale(tomSubsampledInterpBi[:,:,y,:])
xint = 120
xfin = 230
yint = 83
yfin = 183
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot1 = plot1x[xint:xfin,yint:yfin]
miniplot2 = plot2x[xint:xfin,yint:yfin]
# miniplot3 = plot3x[xintsub:xfinsub,yintsub:yfinsub]
miniplot3 = plot3x[xint:xfin,yint:yfin]
miniplot4 = plot4x[xint:xfin,yint:yfin]
ash = np.round(sharpness(miniplot1),decimals=2)
bsh = np.round(sharpness(miniplot2),decimals=2)
csh = np.round(sharpness(miniplot3),decimals=2)
dsh = np.round(sharpness(miniplot4),decimals=2)
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot2, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[1].imshow(miniplot1, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[2].imshow(miniplot3, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot4, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered y={y}_zint{xint}_zfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

#%%

xint = 549
xfin = 649
yint = 573
yfin = 773
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot12 = plot1[xint:xfin,yint:yfin]
miniplot22 = plot2[xint:xfin,yint:yfin]
# miniplot32 = plot3[xintsub:xfinsub,yintsub:yfinsub]
miniplot32 = plot3[xint:xfin,yint:yfin]
miniplot42 = plot4[xint:xfin,yint:yfin]
ash = np.round(sharpness(miniplot1),decimals=2)
bsh = np.round(sharpness(miniplot2),decimals=2)
csh = np.round(sharpness(miniplot3),decimals=2)
dsh = np.round(sharpness(miniplot4),decimals=2)
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot22, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[1].imshow(miniplot12, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[2].imshow(miniplot32, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot42, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%
xint = 100
xfin = 350
yint = 0
yfin = 330
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot13 = axFiltered[xint:xfin,yint:yfin]
miniplot23 = bxFiltered[xint:xfin,yint:yfin]
# miniplot33 = cxFiltered[xintsub:xfinsub,yintsub:yfinsub]
miniplot33 = cxFiltered[xint:xfin,yint:yfin]
miniplot43 = dxFiltered[xint:xfin,yint:yfin]
ash = np.round(sharpness(miniplot1),decimals=2)
bsh = np.round(sharpness(miniplot2),decimals=2)
csh = np.round(sharpness(miniplot3),decimals=2)
dsh = np.round(sharpness(miniplot4),decimals=2)
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot23, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[1].imshow(miniplot13, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[2].imshow(miniplot33, cmap='gray',vmin=vmin2,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot43, cmap='gray',vmin=vmin2,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered x={x}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

xint = 250
xfin = 370
yint = 300
yfin = 600
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot13 = axFiltered[xint:xfin,yint:yfin]
miniplot23 = bxFiltered[xint:xfin,yint:yfin]
# miniplot33 = cxFiltered[xintsub:xfinsub,yintsub:yfinsub]
miniplot33 = cxFiltered[xint:xfin,yint:yfin]
miniplot43 = dxFiltered[xint:xfin,yint:yfin]
ash = np.round(sharpness(miniplot1),decimals=2)
bsh = np.round(sharpness(miniplot2),decimals=2)
csh = np.round(sharpness(miniplot3),decimals=2)
dsh = np.round(sharpness(miniplot4),decimals=2)
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot23, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[1].imshow(miniplot13, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[2].imshow(miniplot33, cmap='gray',vmin=vmin2,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot43, cmap='gray',vmin=vmin2,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered x={x}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%


pathCmap = r'C:\Users\USER\Documents\GitHub\DLOCT'
file = 'c3_colormap.csv'
c3 = pd.read_csv(os.path.join(pathCmap,file),sep=' ',header=None)
custom_cmap = mcolors.ListedColormap(np.array(c3))

color_plot = custom_cmap
vmin = -3
vmax = 3
correlations = []
fileNames = []
enfaceReconstructed = tomReconstructed[z,:,:,:]
correlationReconstructedx,correlationReconstructedy = Correlation(enfaceReconstructed)
stdxr = np.std(correlationReconstructedx)
meanxr = np.mean(correlationReconstructedx)
stdyr = np.std(correlationReconstructedy)
meanyr = np.mean(correlationReconstructedy)
filenamex = f'correlationx_Reconstructed_z={z}_mean={meanxr}_std={stdxr}.svg'
filenamey = f'correlationy_Reconstructed_z={z}_mean={meanyr}_std={stdyr}.svg'
correlations.append(correlationReconstructedx)
correlations.append(correlationReconstructedy)
fileNames.append(filenamex)
fileNames.append(filenamey)

enfaceOriginal = tomOriginal[z,:,:,:]
correlationOriginalx,correlationOriginaly = Correlation(enfaceOriginal)
stdxo = np.std(correlationOriginalx)
meanxo = np.mean(correlationOriginalx)
stdyo = np.std(correlationOriginaly)
meanyo = np.mean(correlationOriginaly)
filenamex = f'correlationx_original_z={z}_mean={meanxo}_std={stdxo}.svg'
filenamey = f'correlationy_original_z={z}_mean={meanyo}_std={stdyo}.svg'
correlations.append(correlationOriginalx)
correlations.append(correlationOriginaly)
fileNames.append(filenamex)
fileNames.append(filenamey)

enfaceSubsampled = tomSubsampledInterpBi[z,:,:,:]
correlationSubsampledx,correlationSubsampledy = Correlation(enfaceSubsampled)
stdxs = np.std(correlationSubsampledx)
meanxs = np.mean(correlationSubsampledx)
stdys = np.std(correlationSubsampledy)
meanys = np.mean(correlationSubsampledy)
filenamex = f'correlationx_linearinterp_z={z}_mean={meanxs}_std={stdxs}.svg'
filenamey = f'correlationy_linearinterp_z={z}_mean={meanys}_std={stdys}.svg'
correlations.append(correlationSubsampledx)
correlations.append(correlationSubsampledy)
fileNames.append(filenamex)
fileNames.append(filenamey)

enfaceInterpolated = tomSubsampledInterp[z,:,:,:]
correlationInterpolatedx,correlationInterpolatedy = Correlation(enfaceInterpolated)
stdxi = np.std(correlationInterpolatedx)
meanxi = np.mean(correlationInterpolatedx)
stdyi = np.std(correlationInterpolatedy)
meanyi = np.mean(correlationInterpolatedy)
filenamex = f'correlationx_biinterpolated_z={z}_mean={meanxi}_std={stdxi}.svg'
filenamey = f'correlationy_biinterpolated_z={z}_mean={meanyi}_std={stdyi}.svg'
correlations.append(correlationInterpolatedx)
correlations.append(correlationInterpolatedy)
fileNames.append(filenamex)
fileNames.append(filenamey)


enfaceSubsampled2 = tomSubsampled[z,:,:,:]
correlationSubsampledx2,correlationSubsampledy2 = Correlation(enfaceSubsampled2)



fig, axs = plt.subplots(1, 4, figsize=(20, 5))
cmap= plt.cm.hsv
norm = cmnorm(vmin=-3, vmax=3)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(correlationOriginaly,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0].axis('off')
axs[0].set_title(f'Original mean= {meanyo}')

axs[1].imshow(correlationReconstructedy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed mean= {meanyr}')

axs[2].imshow(correlationInterpolatedy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanyi}')

axs[3].imshow(correlationSubsampledy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[3].axis('off')
axs[3].set_title(f'Subsampled mean= {meanys}')
plt.subplots_adjust(wspace=0.05, hspace=0)

figname = f'Phase correlation axis y z={z}.png'
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='Phase')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=150,format='svg')
    print('fig saved')
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
cmap= plt.cm.hsv
norm = cmnorm(vmin=-3, vmax=3)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(correlationOriginalx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0].axis('off')
axs[0].set_title(f'Original mean= {meanxo}')

axs[1].imshow(correlationReconstructedx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed mean= {meanxr}')

axs[2].imshow(correlationInterpolatedx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanxi}')

axs[3].imshow(correlationSubsampledx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[3].axis('off')
axs[3].set_title(f'Subsampled mean= {meanxs}')
plt.subplots_adjust(wspace=0.05, hspace=0)

figname = f'Phase correlation axis x z={z}.png'
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='Phase')
if savefig:  
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=150,format='svg')
    print('fig saved')
plt.show()


fig, axs = plt.subplots(2, 4, figsize=(20, 10))
cmap= plt.cm.hsv
norm = cmnorm(vmin=-3, vmax=3)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0,0].imshow(correlationOriginalx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0,0].axis('off')
# axs[0,0].set_title(f'Original mean= {meanxo}')
axs[0,0].set_title(f'Original')

axs[0,1].imshow(correlationReconstructedx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0,1].axis('off') 
# axs[0,1].set_title(f'cGAN reconstructed mean= {meanxr}')
axs[0,1].set_title(f'cGAN Reconstructed')

axs[0,2].imshow(correlationInterpolatedx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0,2].axis('off')
# axs[0,2].set_title(f'Subsampled interpolated mean= {meanxi}')
axs[0,2].set_title(f'Subsampled Interpolated')

axs[0,3].imshow(correlationSubsampledx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0,3].axis('off')
# axs[0,3].set_title(f'Subsampled mean= {meanxs}')
axs[0,3].set_title(f'Subsampled')

axs[1,0].imshow(correlationOriginaly,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1,0].axis('off')
# axs[1,0].set_title(f'Original mean= {meanyo}')

axs[1,1].imshow(correlationReconstructedy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1,1].axis('off') 
# axs[1,1].set_title(f'cGAN reconstructed mean= {meanyr}')

axs[1,2].imshow(correlationInterpolatedy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1,2].axis('off')
# axs[1,2].set_title(f'Subsampled interpolated mean= {meanyi}')

axs[1,3].imshow(correlationSubsampledy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1,3].axis('off')
# axs[1,3].set_title(f'Subsampled mean= {meanys}')

plt.subplots_adjust(wspace=0.01, hspace=0.01)

figname = f'Phase correlation z={z}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[1,3], label='Phase')
if savefig:  
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=150,format='svg')
    print('fig saved')
plt.show()

minicorrelations = []
xinty = 430
xfiny = 630
yinty = 140
yfiny = 340

xintx = 430
xfinx = 630
yintx = 140
yfinx = 340

minicorrreconsty = correlationReconstructedy[xinty:xfiny,yinty:yfiny]
minicorrreconstx = correlationReconstructedx[xintx:xfinx,yintx:yfinx]
minicorrelations.append(minicorrreconstx)
minicorrelations.append(minicorrreconsty)
minicorroriginaly = correlationOriginaly[xinty:xfiny,yinty:yfiny]
minicorroriginalx = correlationOriginalx[xintx:xfinx,yintx:yfinx]
minicorrelations.append(minicorroriginalx)
minicorrelations.append(minicorroriginaly)
minicorrlineary = correlationSubsampledy[xinty:xfiny,yinty:yfiny]
minicorrlinearx = correlationSubsampledx[xintx:xfinx,yintx:yfinx]
minicorrelations.append(minicorrlinearx)
minicorrelations.append(minicorrlineary)
minicorrcubicy = correlationInterpolatedy[xinty:xfiny,yinty:yfiny]
minicorrcubicx = correlationInterpolatedx[xintx:xfinx,yintx:yfinx]
minicorrelations.append(minicorrcubicx)
minicorrelations.append(minicorrcubicy)

#%%

if savefigindividuals:
    for i in tqdm(range(len(fileNames))):
        
        image = correlations[i]
        figname = fileNames[i]
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=color_plot,vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(path,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()

        image = minicorrelations[i]
        minifigname = f'roi_x={xinty}...{xfiny}_y={yinty}...{yfiny}_{fileNames[i]}'
        print(figname)
        # if i == 1 or i == 3 :
        #     minifigname = f'roi_x={xinty}...{xfiny}_y={yinty}...{yfiny}_{figname}'
        # elif i==0 or i== 2:
        #     minifigname = f'roi_x={xintx}...{xfinx}_y={yintx}...{yfinx}_{figname}'
        print(minifigname)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=color_plot,vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(path,folder,subfolder,minifigname),
                     bbox_inches='tight', pad_inches=0, dpi=100,format='svg')
        plt.close()


fig, axs = plt.subplots(1, 4, figsize=(30, 5))

axs[0].imshow(minicorroriginaly,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0].axis('off')
axs[0].set_title(f'Original mean= {meanxo}')

axs[1].imshow(minicorrreconsty,vmax= vmax, vmin=vmin, cmap=color_plot,aspect='auto')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed mean= {meanxr}')

axs[2].imshow(minicorrlineary,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanxi}')

axs[3].imshow(minicorrcubicy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[3].axis('off')
axs[3].set_title(f'Subsampled mean= {meanxs}')
plt.subplots_adjust(wspace=0.05, hspace=0)

figname = f'zoom Phase correlation axis y z={z}.png'
if savefig:  
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=150,format='svg')
    print('fig saved')
plt.show()



fig, axs = plt.subplots(1, 4, figsize=(30, 5))
vmin = -3
vmax = 3
axs[0].imshow(minicorroriginalx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0].axis('off')
axs[0].set_title(f'Original mean= {meanxo}')

axs[1].imshow(minicorrreconstx,vmax= vmax, vmin=vmin, cmap=color_plot,aspect='auto')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed mean= {meanxr}')

axs[2].imshow(minicorrlinearx,cmap=color_plot,aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanxi}')

axs[3].imshow(minicorrcubicx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[3].axis('off')
axs[3].set_title(f'Subsampled mean= {meanxs}')
plt.subplots_adjust(wspace=0.05, hspace=0)

figname = f'zoom Phase correlation axis x z={z}.png'
if savefig:  
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=150,format='svg')
    print('fig saved')
plt.show()

#%%
mpsoriginal,_ = Powerspectrum(enfaceOriginal[:,:,0]+1j*enfaceOriginal[:,:,1])
mpsreconstructed,mpsreconstructedfullscale = Powerspectrum(enfaceReconstructed[:,:,0]+1j*enfaceReconstructed[:,:,1])
mpslinear,_ = Powerspectrum(enfaceSubsampled[:,:,0]+1j*enfaceSubsampled[:,:,1])
mpscubic,_ = Powerspectrum(enfaceInterpolated[:,:,0]+1j*enfaceInterpolated[:,:,1])



#%%
mpsoriginal = MPS_single(enfaceOriginal[:,:,0]+1j*enfaceOriginal[:,:,1],meandim=0)
mpsreconstructed= MPS_single(enfaceReconstructed[:,:,0]+1j*enfaceReconstructed[:,:,1],meandim=0)
mpslinear = MPS_single(enfaceSubsampled[:,:,0]+1j*enfaceSubsampled[:,:,1],meandim=0)
mpscubic = MPS_single(enfaceInterpolated[:,:,0]+1j*enfaceInterpolated[:,:,1],meandim=0)

#%%
promedios_reales = np.mean(enfaceSubsampled[:,:,0], axis=1)
promedios_imaginarios = np.mean(enfaceSubsampled[:,:,1], axis=1)

# # Graficar los promedios a lo largo del eje y
# plt.figure(figsize=(10, 6))
# plt.plot(promedios_reales, label='Promedio Real')
# plt.plot(promedios_imaginarios, label='Promedio Imaginario')
# plt.xlabel('Índice de Fila (Eje Y)')
# plt.ylabel('Promedio del Valor de los Píxeles')
# plt.title('Promedios de los Valores en el Eje Y para Cada Canal')
# plt.legend()
# plt.grid(True)
# plt.show()


#%%
tomSubsampled2 = tomOriginal[:,:,1::2,:]#%%
promedios_reales = np.mean(tomSubsampled2[0,:,:,0], axis=1)
promedios_imaginarios = np.mean(tomSubsampled2[0,:,:,1], axis=1)

#%%
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmsrx = rmse(minicorroriginalx,minicorrreconstx)
rmsry = rmse(minicorroriginaly,minicorrreconsty)
print(rmsrx)
print(rmsry)

rmsix = rmse(minicorroriginalx,minicorrlinearx)
rmsiy = rmse(minicorroriginaly,minicorrlineary)

print(rmsix)
print(rmsiy)

rmscx = rmse(minicorroriginalx,minicorrcubicx)
rmscy = rmse(minicorroriginaly,minicorrcubicy)

print(rmscx)
print(rmscy)