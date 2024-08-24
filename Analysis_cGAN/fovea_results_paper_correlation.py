#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
from numpy.fft import fft2,fft,fftshift
from Deep_Utils import MPS_single, Powerspectrum,dbscale,Correlation
from Utils import downSampleSlices,downSampleSlicesInterp
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
#%%
tomSubsampled = tomOriginal[:,:,0::2,:]
print('tom Subsampled')
#%%

tomSubsampledInterp = np.zeros((nZbin,nXbin,nYbin,2))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    tomSubsampledInterp[z,:,:,:] = cv2.resize(tomSubsampled[z,:,:,:],
                dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_NEAREST)
print('tom linearlly interpolated')

tomSubsampledInterpBi = np.zeros((nZbin, nXbin, nYbin, 2))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    tomSubsampledInterpBi[z, :, :, :] = cv2.resize(
        tomSubsampled[z, :, :, :], 
        dsize=(int(np.shape(tomSubsampled)[2]*2), np.shape(tomSubsampled)[1]),
        interpolation=cv2.INTER_CUBIC)
print('tom cubic interpolated')
#%%
z = 170
x = 519
folder = 'cortes'
subfolder = f'corte{6}'
savefig = False
savefigindividuals = True

pathCmap = r'C:\Users\USER\Documents\GitHub\DLOCT'
file = 'c3_colormap.csv'
c3 = pd.read_csv(os.path.join(pathCmap,file),sep=' ',header=None)
custom_cmap = mcolors.ListedColormap(np.array(c3))
colors = ['rgb({}, {}, {})'.format(int(r), int(g), int(b)) for r, g, b in np.array(c3)*255]
#%%

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

enfaceSubsampledLin = tomSubsampledInterp[z,:,:,:]
correlationLinx,correlationLiny = Correlation(enfaceSubsampledLin)
stdxs = np.std(correlationLinx)
meanxs = np.mean(correlationLinx)
stdys = np.std(correlationLiny)
meanys = np.mean(correlationLiny)
filenamex = f'correlationx_linearinterp_z={z}_mean={meanxs}_std={stdxs}.svg'
filenamey = f'correlationy_linearinterp_z={z}_mean={meanys}_std={stdys}.svg'
correlations.append(correlationLinx)
correlations.append(correlationLiny)
fileNames.append(filenamex)
fileNames.append(filenamey)

enfaceSubsampledBi = tomSubsampledInterpBi[z,:,:,:]
correlationBix,correlationBiy = Correlation(enfaceSubsampledBi)
stdxi = np.std(correlationBix)
meanxi = np.mean(correlationBix)
stdyi = np.std(correlationBiy)
meanyi = np.mean(correlationBiy)
filenamex = f'correlationx_biinterpolated_z={z}_mean={meanxi}_std={stdxi}.svg'
filenamey = f'correlationy_biinterpolated_z={z}_mean={meanyi}_std={stdyi}.svg'
correlations.append(correlationBix)
correlations.append(correlationBiy)
fileNames.append(filenamex)
fileNames.append(filenamey)

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

axs[2].imshow(correlationLiny,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanyi}')

axs[3].imshow(correlationBiy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
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

axs[2].imshow(correlationLinx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[2].axis('off')
axs[2].set_title(f'Subsampled interpolated mean= {meanxi}')

axs[3].imshow(correlationBix,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
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

axs[0,2].imshow(correlationLinx,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0,2].axis('off')
# axs[0,2].set_title(f'Subsampled interpolated mean= {meanxi}')
axs[0,2].set_title(f'Subsampled Interpolated')

axs[0,3].imshow(correlationBix,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[0,3].axis('off')
# axs[0,3].set_title(f'Subsampled mean= {meanxs}')
axs[0,3].set_title(f'Subsampled')

axs[1,0].imshow(correlationOriginaly,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1,0].axis('off')
# axs[1,0].set_title(f'Original mean= {meanyo}')

axs[1,1].imshow(correlationReconstructedy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1,1].axis('off') 
# axs[1,1].set_title(f'cGAN reconstructed mean= {meanyr}')

axs[1,2].imshow(correlationLiny,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
axs[1,2].axis('off')
# axs[1,2].set_title(f'Subsampled interpolated mean= {meanyi}')

axs[1,3].imshow(correlationBiy,vmax= vmax, vmin=vmin,cmap=color_plot,aspect='auto')
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
minicorrlineary = correlationLiny[xinty:xfiny,yinty:yfiny]
minicorrlinearx = correlationLinx[xintx:xfinx,yintx:yfinx]
minicorrelations.append(minicorrlinearx)
minicorrelations.append(minicorrlineary)
minicorrcubicy = correlationBiy[xinty:xfiny,yinty:yfiny]
minicorrcubicx = correlationBix[xintx:xfinx,yintx:yfinx]
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
promedios_reales = np.mean(tomSubsampled[z,:,:,0], axis=1)
promedios_imaginarios = np.mean(tomSubsampled[z,:,:,1], axis=1)

# Graficar los promedios a lo largo del eje y
plt.figure(figsize=(10, 6))
plt.plot(promedios_reales, label='Promedio Real')
plt.plot(promedios_imaginarios, label='Promedio Imaginario')
plt.xlabel('Índice de Fila (Eje Y)')
plt.ylabel('Promedio del Valor de los Píxeles')
plt.title('Promedios de los Valores en el Eje Y para Cada Canal')
plt.legend()
plt.grid(True)
plt.show()

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

#%%

# corrx,corry = Correlation(tomSubsampled[z,:,:,:])
# fig = px.imshow(
#     corry,
#     color_continuous_scale=colors,
#     zmin=-np.pi,
#     zmax=np.pi
# )

# # Quitar etiquetas de los ejes
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)

# # Quitar la barra de color
# fig.update(layout_coloraxis_showscale=False)

# # Hacer el fondo transparente
# fig.update_layout(
#     plot_bgcolor='rgba(0,0,0,0)',  # Fondo de la gráfica
#     paper_bgcolor='rgba(0,0,0,0)'  # Fondo del papel
# )

# # Guardar la figura en formato SVG
# fig.write_image("figura3.svg", format='svg')