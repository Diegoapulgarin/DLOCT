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
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..480)_subsampled'
nYbins = int(nYbin/2)
tom = np.fromfile(path+'\\'+filename+real,'single')
tom = np.reshape(tom,(nZbin,nXbin,nYbins,npol),order='F')
tom = np.sum(tom,axis=3)
tomi = np.fromfile(path+'\\'+filename+imag,'single')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbins,npol),order='F')
tomi = np.sum(tomi,axis=3)
tomSubsampled = np.stack((tom, tomi), axis=3)
del tom, tomi
print('subsampled loaded')
#%%
tomSubsampledInterp = np.zeros((nZbin,nXbin,nYbin,2))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    tomSubsampledInterp[z,:,:,0] = cv2.resize(tomSubsampled[z,:,:,0],
                dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_LINEAR)
    tomSubsampledInterp[z,:,:,1] = cv2.resize(tomSubsampled[z,:,:,1],
                dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_LINEAR)
#%%
z = 170
x = 519
folder = 'cortes'
subfolder = f'corte{6}'
savefig = False
savefigindividuals = False
vmin= 70
vmax = 120

plot1x = dbscale(tomReconstructed[:,x,:,:])
plot2x = dbscale(tomOriginal[:,x,:,:])
plot3x = dbscale(tomSubsampled[:,x,:,:])
plot4x = dbscale(tomSubsampledInterp[:,x,:,:])

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

axs[2].imshow(plot4x, cmap='gray',vmin=vmin,vmax=vmax, aspect = 'auto')
axs[2].axis('off')
axs[2].set_title('Subsampled interpolated')

axs[3].imshow(plot3x, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
axs[3].set_title('Subsampled')

plt.subplots_adjust(wspace=0.01, hspace=0)
figname = f'comparision x={x}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB') 
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
 
plt.show()

plot1 = dbscale(tomReconstructed[z,:,:,:])
plot2 = dbscale(tomOriginal[z,:,:,:])
plot3 = dbscale(tomSubsampled[z,:,:,:])
plot4 = dbscale(tomSubsampledInterp[z,:,:,:])

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

axs[2].imshow(plot4, cmap='gray',vmin=vmin,vmax=vmax, aspect = 'auto')
axs[2].axis('off')
axs[2].set_title('Subsampled interpolated')

axs[3].imshow(plot3, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
axs[3].set_title('Subsampled')

plt.subplots_adjust(wspace=0.01, hspace=0)
figname = f'comparision z={z}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB') 
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
 
plt.show()


aFiltered = non_local_means_despeckling_3d(plot1,search_window_size=21, block_size=5)
bFiltered = non_local_means_despeckling_3d(plot2,search_window_size=21, block_size=5)
cFiltered = non_local_means_despeckling_3d(plot3,search_window_size=21, block_size=5)
dFiltered = non_local_means_despeckling_3d(plot4,search_window_size=21, block_size=5)

axFiltered = non_local_means_despeckling_3d(plot1x,search_window_size=21, block_size=5)
bxFiltered = non_local_means_despeckling_3d(plot2x,search_window_size=21, block_size=5)
cxFiltered = non_local_means_despeckling_3d(plot3x,search_window_size=21, block_size=5)
dxFiltered = non_local_means_despeckling_3d(plot4x,search_window_size=21, block_size=5)

xint = 100
xfin = 323
yint = 80
yfin = 323
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot1 = aFiltered[xint:xfin,yint:yfin]
miniplot2 = bFiltered[xint:xfin,yint:yfin]
miniplot3 = cFiltered[xintsub:xfinsub,yintsub:yfinsub]
miniplot4 = dFiltered[xint:xfin,yint:yfin]
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

axs[2].imshow(miniplot4, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot3, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

xint = 549
xfin = 649
yint = 573
yfin = 773
xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
miniplot12 = aFiltered[xint:xfin,yint:yfin]
miniplot22 = bFiltered[xint:xfin,yint:yfin]
miniplot32 = cFiltered[xintsub:xfinsub,yintsub:yfinsub]
miniplot42 = dFiltered[xint:xfin,yint:yfin]
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

axs[2].imshow(miniplot42, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot32, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

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
miniplot33 = cxFiltered[xintsub:xfinsub,yintsub:yfinsub]
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

axs[2].imshow(miniplot43, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot33, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
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
miniplot33 = cxFiltered[xintsub:xfinsub,yintsub:yfinsub]
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

axs[2].imshow(miniplot43, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[2].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[3].imshow(miniplot33, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[3].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0)

figname = f'comparision filtered x={x}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

correlations = []
fileNames = []

enfaceReconstructed = tomReconstructed[z,:,:,:]
correlationReconstructedx,correlationReconstructedy = Correlation(enfaceReconstructed)
stdxr = np.std(correlationReconstructedx)
meanxr = np.mean(correlationReconstructedx)
stdyr = np.std(correlationReconstructedy)
meanyr = np.mean(correlationReconstructedy)
filenamex = f'correlationx_Reconstructed_z={z}_mean={meanxr}_std={stdxr}.png'
filenamey = f'correlationy_Reconstructed_z={z}_mean={meanyr}_std={stdyr}.png'
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
filenamex = f'correlationx_original_z={z}_mean={meanxo}_std={stdxo}.png'
filenamey = f'correlationy_original_z={z}_mean={meanyo}_std={stdyo}.png'
correlations.append(correlationOriginalx)
correlations.append(correlationOriginaly)
fileNames.append(filenamex)
fileNames.append(filenamey)

enfaceSubsampled = tomSubsampled[z,:,:,:]
correlationSubsampledx,correlationSubsampledy = Correlation(enfaceSubsampled)
stdxs = np.std(correlationSubsampledx)
meanxs = np.mean(correlationSubsampledx)
stdys = np.std(correlationSubsampledy)
meanys = np.mean(correlationSubsampledy)
filenamex = f'correlationx_subsampled_z={z}_mean={meanxs}_std={stdxs}.png'
filenamey = f'correlationy_subsampled_z={z}_mean={meanys}_std={stdys}.png'
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
filenamex = f'correlationx_interpolated_z={z}_mean={meanxi}_std={stdxi}.png'
filenamey = f'correlationy_interpolated_z={z}_mean={meanyi}_std={stdyi}.png'
correlations.append(correlationInterpolatedx)
correlations.append(correlationInterpolatedy)
fileNames.append(filenamex)
fileNames.append(filenamey)


fig, axs = plt.subplots(1, 4, figsize=(20, 5))
cmap= plt.cm.twilight
norm = cmnorm(vmin=-3, vmax=3)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(correlationOriginaly,cmap='twilight',aspect='auto')
axs[0].axis('off')
axs[0].set_title(f'Original mean= {meanyo}')

axs[1].imshow(correlationReconstructedy,cmap='twilight',aspect='auto')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed mean= {meanyr}')

axs[2].imshow(correlationInterpolatedy,cmap='twilight',aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanyi}')

axs[3].imshow(correlationSubsampledy,cmap='twilight',aspect='auto')
axs[3].axis('off')
axs[3].set_title(f'Subsampled mean= {meanys}')
plt.subplots_adjust(wspace=0.05, hspace=0)

figname = f'Phase correlation axis y z={z}.png'
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='Phase')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
cmap= plt.cm.twilight
norm = cmnorm(vmin=-3, vmax=3)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(correlationOriginalx,cmap='twilight',aspect='auto')
axs[0].axis('off')
axs[0].set_title(f'Original mean= {meanxo}')

axs[1].imshow(correlationReconstructedx,cmap='twilight',aspect='auto')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed mean= {meanxr}')

axs[2].imshow(correlationInterpolatedx,cmap='twilight',aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanxi}')

axs[3].imshow(correlationSubsampledx,cmap='twilight',aspect='auto')
axs[3].axis('off')
axs[3].set_title(f'Subsampled mean= {meanxs}')
plt.subplots_adjust(wspace=0.05, hspace=0)

figname = f'Phase correlation axis x z={z}.png'
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='Phase')
if savefig:  
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%

fig, axs = plt.subplots(2, 4, figsize=(20, 10))
cmap= plt.cm.twilight
norm = cmnorm(vmin=-3, vmax=3)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0,0].imshow(correlationOriginalx,cmap='twilight',aspect='auto')
axs[0,0].axis('off')
# axs[0,0].set_title(f'Original mean= {meanxo}')
axs[0,0].set_title(f'Original')

axs[0,1].imshow(correlationReconstructedx,cmap='twilight',aspect='auto')
axs[0,1].axis('off') 
# axs[0,1].set_title(f'cGAN reconstructed mean= {meanxr}')
axs[0,1].set_title(f'cGAN Reconstructed')

axs[0,2].imshow(correlationInterpolatedx,cmap='twilight',aspect='auto')
axs[0,2].axis('off')
# axs[0,2].set_title(f'Subsampled interpolated mean= {meanxi}')
axs[0,2].set_title(f'Subsampled Interpolated')

axs[0,3].imshow(correlationSubsampledx,cmap='twilight',aspect='auto')
axs[0,3].axis('off')
# axs[0,3].set_title(f'Subsampled mean= {meanxs}')
axs[0,3].set_title(f'Subsampled')

axs[1,0].imshow(correlationOriginaly,cmap='twilight',aspect='auto')
axs[1,0].axis('off')
# axs[1,0].set_title(f'Original mean= {meanyo}')

axs[1,1].imshow(correlationReconstructedy,cmap='twilight',aspect='auto')
axs[1,1].axis('off') 
# axs[1,1].set_title(f'cGAN reconstructed mean= {meanyr}')

axs[1,2].imshow(correlationInterpolatedy,cmap='twilight',aspect='auto')
axs[1,2].axis('off')
# axs[1,2].set_title(f'Subsampled interpolated mean= {meanyi}')

axs[1,3].imshow(correlationSubsampledy,cmap='twilight',aspect='auto')
axs[1,3].axis('off')
# axs[1,3].set_title(f'Subsampled mean= {meanys}')

plt.subplots_adjust(wspace=0.01, hspace=0.01)

figname = f'Phase correlation z={z}.png'
# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[1,3], label='Phase')
if savefig:  
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()


if savefigindividuals:
    for i in tqdm(range(len(fileNames))):
        image = correlations[i]
        figname = fileNames[i]
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='twilight')  # Puedes cambiar 'viridis' por el colormap que prefieras.
        # Elimina los ejes y bordes blancos
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # Guarda la imagen
        plt.savefig(os.path.join(path,folder,subfolder,figname), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()