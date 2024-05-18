'''
Z depth = 3.9 um for each pixel
X depth = 14 um for each pixel
Y depth = 28 um for each pixel
'''
#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
from Deep_Utils import Correlation, MPS_single, Powerspectrum, calculate_mse
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import scipy
from matplotlib.colors import Normalize as cmnorm
from matplotlib.cm import ScalarMappable
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm
CentralWavelength = 870e-9
bandwith = 50e-9
pixel = (2*np.log(2)/np.pi)*(CentralWavelength**2/bandwith)
#%%
path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
# filename = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomoriginalFlat_z=400_x=896_y=960_pol=2'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 586
# nZbin = 400
nXbin = 896
nYbin = 960
npol = 2
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,npol),order='F')
tom = np.sum(tom,axis=3)
tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,npol),order='F')
tomi = np.sum(tomi,axis=3)
tomOriginal = np.stack((tom, tomi), axis=3)
del tom, tomi
print('original loaded')
# filename = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomFlat_z=400_x=896_y=960_pol=2'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed'
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbin),order='F')
# tom = np.sum(tom,axis=3)
tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin),order='F')
# tomi = np.sum(tomi,axis=3)
tomReconstructed = np.stack((tom, tomi), axis=3)
del tom, tomi
print('reconstructed loaded')
# filename = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomSubsampledFlat_z=400_x=896_y=480_pol=2'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..480)_subsampled'
nYbins = int(nYbin/2)
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbins,npol),order='F')
tom = np.sum(tom,axis=3)
tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbins,npol),order='F')
tomi = np.sum(tomi,axis=3)
tomSubsampled = np.stack((tom, tomi), axis=3)
del tom, tomi
print('subsampled loaded')
#%%
def sharpness(a):
    b = scipy.ndimage.gaussian_laplace(a,sigma=3)
    c = (b-np.min(b))/(np.max(b)-np.min(b))
    sharpnessValue = c.sum()
    return sharpnessValue
def non_local_means_despeckling_3d(volume, h=None, search_window_size=21, block_size=5):
    if h is None:
        # If 'h' is not provided, estimate it from the input volume
        sigma_estimated = estimate_sigma(volume, average_sigmas=True)
        h = 0.8 * sigma_estimated
        
    # Apply the 3D Non-local Means filter
    despeckled_volume = denoise_nl_means(volume, h=h, fast_mode=True, patch_size=block_size, patch_distance=search_window_size)

    return despeckled_volume
# print(sharpness(aFiltered),sharpness(bFiltered),sharpness(cFiltered))
#%%
from tqdm import tqdm
subsampled2 = np.zeros((586,896,960))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    subsampled2[z,:,:] = cv2.resize(abs(tomSubsampled[z,:,:,0]+1j*tomSubsampled[z,:,:,1])**2,
                dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_LINEAR)
#%%
subsampledReal = np.zeros((586,896,960))
subsampledImag = np.zeros((586,896,960))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    subsampledReal[z,:,:] = cv2.resize(tomSubsampled[z,:,:,0],
                dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_LINEAR)
    subsampledImag[z,:,:] = cv2.resize(tomSubsampled[z,:,:,1],
                dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_LINEAR)
subsampled2 = subsampledReal+1j*subsampledImag
del subsampledReal, subsampledImag
#%%
z = 170
a = abs(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1])**2
b = abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2
c = abs(subsampled2[z,:,:])**2
# print(sharpness(a),sharpness(b),sharpness(c))
aFiltered = non_local_means_despeckling_3d(a,search_window_size=21, block_size=5)
bFiltered = non_local_means_despeckling_3d(b,search_window_size=21, block_size=5)
cFiltered = non_local_means_despeckling_3d(c,search_window_size=21, block_size=5)
#%%
# res = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
z = 170
a = abs(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1])**2
b = abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2
c = abs(subsampled2[z,:,:])**2
vmin= 70
vmax = 120
plot1 = 10*np.log10(a)
plot2 = 10*np.log10(b)
plot3 = 10*np.log10(c)
# plt.imshow(plot3, cmap='gray',vmin=60,vmax=120)
# image = cv2.GaussianBlur(image, (3, 3), 0)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(plot2, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[0].axis('off')
axs[0].set_title('Original')

axs[1].imshow(plot1, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[1].axis('off') 
axs[1].set_title('cGAN reconstructed')

axs[2].imshow(plot3, cmap='gray',vmin=vmin,vmax=vmax, aspect = 'equal')
axs[2].axis('off')
axs[2].set_title('Subsampled')

plt.subplots_adjust(wspace=0.05, hspace=0)
figname = f'comparision z={z}.png'
# plt.savefig(os.path.join(path,figname), dpi=300)
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[2], label='dB')  
plt.show()
#%%

vmin= 50
vmax = 120
plot1 = 10*np.log10(abs(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1])**2)
plot2 = 10*np.log10(abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2)
plot3 = 10*np.log10(abs(tomSubsampled[z,:,:,0]+1j*tomSubsampled[z,:,:,1])**2)
# plt.imshow(plot3, cmap='gray',vmin=60,vmax=120)
# image = cv2.GaussianBlur(image, (3, 3), 0)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(plot2, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[0].axis('off')
axs[0].set_title('Original')

axs[1].imshow(plot1, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[1].axis('off') 
axs[1].set_title('cGAN reconstructed')

axs[2].imshow(plot3, cmap='gray',vmin=vmin,vmax=vmax,interpolation='none', extent=[95,130,32,0])
axs[2].axis('off')
axs[2].set_title('Subsampled')

plt.subplots_adjust(wspace=0.05, hspace=0)
figname = f'comparision z={z}.png'
# plt.savefig(os.path.join(path,figname), dpi=300)
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[2], label='dB')  
plt.show()

#%% filtered
xint = 381
xfin = 400
yint = 154
yfin = 194


xintsub = xint
xfinsub = xfin
yintsub = int(yint/2)
yfinsub = int(yfin/2)
a = aFiltered[xint:xfin,yint:yfin]
b = bFiltered[xint:xfin,yint:yfin]
# c = cFiltered[xintsub:xfinsub,yintsub:yfinsub]
c = cFiltered[xint:xfin,yint:yfin]
miniplot1z46 = 10*np.log10(a)
miniplot2z46 = 10*np.log10(b)
miniplot3z46 = 10*np.log10(c)
ash = np.round(sharpness(a),decimals=2)
bsh = np.round(sharpness(b),decimals=2)
csh = np.round(sharpness(c),decimals=2)
vmin = 75
vmax = 120
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot2z46, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[0].axis('off')
axs[0].set_title(f'Original sharpness= {bsh}')

axs[1].imshow(miniplot1z46, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed sharpness= {ash}')

axs[2].imshow(miniplot3z46, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[2].axis('off')
axs[2].set_title(f'Subsampled sharpeness= {csh}')

plt.subplots_adjust(wspace=0.05, hspace=0)
folder = 'cortes'
subfolder = f'corte{5}'
figname = f'comparision filtered z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[2], label='dB')  
# plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
plt.show()

print(sharpness(a),sharpness(b),sharpness(c))
#%% no filtered
a = abs(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1])**2
b = abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2
c = abs(tomSubsampled[z,:,:,0]+1j*tomSubsampled[z,:,:,1])**2
a = a[xint:xfin,yint:yfin]
b = b[xint:xfin,yint:yfin]
c = c[xintsub:xfinsub,yintsub:yfinsub]
miniplot1z46 = 10*np.log10(a)
miniplot2z46 = 10*np.log10(b)
miniplot3z46 = 10*np.log10(c)
ash = np.round(sharpness(a),decimals=2)
bsh = np.round(sharpness(b),decimals=2)
csh = np.round(sharpness(c),decimals=2)
vmin = 75
vmax = 120
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot2z46, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[0].axis('off')
axs[0].set_title(f'Original sharpness= {bsh}')

axs[1].imshow(miniplot1z46, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed sharpness= {ash}')

axs[2].imshow(miniplot3z46, cmap='gray',vmin=vmin,vmax=vmax,interpolation='none', extent=[90,120,32,0])
axs[2].axis('off')
axs[2].set_title(f'Subsampled sharpeness= {csh}')

plt.subplots_adjust(wspace=0.05, hspace=0)

figname = f'comparision no filtered z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[2], label='dB')  
plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
plt.show()
print(sharpness(a),sharpness(b),sharpness(c))
#%% contornos
import cv2

def apply_edge_filter(image_array):
    # Convertir la imagen a escala de grises si no lo está
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # Si es una imagen en color (BGR)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Asegurarse de que la imagen sea de tipo uint8
    if image_array.dtype != 'uint8':
        image_array = cv2.convertScaleAbs(image_array)

    # Aplicar el filtro de contornos
    edges = cv2.Canny(image_array, 0, 10)
    return edges

def apply_sobel_filter(image_array):
    # Convertir la imagen a escala de grises si no lo está
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # Si es una imagen en color (BGR)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el filtro de Sobel en la dirección x y y
    sobelx = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=5)

    # Calcular la magnitud del gradiente
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Convertir la magnitud a tipo uint8 para visualización
    sobel_edges = cv2.convertScaleAbs(magnitude)
    
    return sobel_edges

def apply_laplacian_filter(image_array):
    # Convertir la imagen a escala de grises si no lo está
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # Si es una imagen en color (BGR)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Asegurarse de que la imagen sea de tipo uint8
    if image_array.dtype != 'uint8':
        image_array = cv2.convertScaleAbs(image_array)

    # Aplicar el filtro Laplaciano
    laplacian = cv2.Laplacian(image_array, cv2.CV_64F)
    
    # Convertir el resultado a tipo uint8 para visualización
    laplacian_edges = cv2.convertScaleAbs(laplacian)
    
    return laplacian_edges

def combine_sobel_laplacian(image_array):
    sobel_edges = apply_sobel_filter(image_array)
    laplacian_edges = apply_laplacian_filter(image_array)
    
    # Combinar los resultados usando una operación de bitwise OR
    combined_edges = cv2.bitwise_or(sobel_edges, laplacian_edges)
    
    return combined_edges

def apply_scharr_filter(image_array):
    # Convertir la imagen a escala de grises si no lo está
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Asegurarse de que la imagen sea de tipo uint8
    if image_array.dtype != 'uint8':
        image_array = cv2.convertScaleAbs(image_array)

    # Aplicar el filtro de Scharr en la dirección x y y
    scharrx = cv2.Scharr(image_array, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(image_array, cv2.CV_64F, 0, 1)

    # Calcular la magnitud del gradiente
    magnitude = np.sqrt(scharrx**2 + scharry**2)
    
    # Convertir la magnitud a tipo uint8 para visualización
    scharr_edges = cv2.convertScaleAbs(magnitude)
    
    return scharr_edges
def apply_gaussian_scharr_filter(image_array):
    # Convertir la imagen a escala de grises si no lo está
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Asegurarse de que la imagen sea de tipo uint8
    if image_array.dtype != 'uint8':
        image_array = cv2.convertScaleAbs(image_array)
    
    # Aplicar el filtro Gaussiano para eliminar el ruido
    blurred_image = cv2.GaussianBlur(image_array, (5, 5), 0)
    
    # Aplicar el filtro de Scharr en la dirección x y y
    scharrx = cv2.Scharr(blurred_image, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(blurred_image, cv2.CV_64F, 0, 1)

    # Calcular la magnitud del gradiente
    magnitude = np.sqrt(scharrx**2 + scharry**2)
    
    # Convertir la magnitud a tipo uint8 para visualización
    scharr_edges = cv2.convertScaleAbs(magnitude)
    
    return scharr_edges

def apply_gaussian_filter(image_array):
    # Convertir la imagen a escala de grises si no lo está
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Asegurarse de que la imagen sea de tipo uint8
    if image_array.dtype != 'uint8':
        image_array = cv2.convertScaleAbs(image_array)
    
    # Aplicar el filtro Gaussiano para eliminar el ruido
    blurred_image = cv2.GaussianBlur(image_array, (9, 9), 0)
    
    return blurred_image

a = aFiltered[xint:xfin,yint:yfin]
b = bFiltered[xint:xfin,yint:yfin]
c = cFiltered[xintsub:xfinsub,yintsub:yfinsub]
miniplot1z46 = apply_scharr_filter(apply_gaussian_filter(10*np.log10(a)))
miniplot2z46 = apply_scharr_filter(apply_gaussian_filter(10*np.log10(b)))
miniplot3z46 = apply_scharr_filter(apply_gaussian_filter(10*np.log10(c)))
ash = np.round(sharpness(a),decimals=2)
bsh = np.round(sharpness(b),decimals=2)
csh = np.round(sharpness(c),decimals=2)
vmin = 0
vmax = 120
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0].imshow(miniplot2z46, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[0].axis('off')
axs[0].set_title(f'Original sharpness= {bsh}')

axs[1].imshow(miniplot1z46, cmap='gray',vmin=vmin,vmax=vmax,aspect='equal')
axs[1].axis('off') 
axs[1].set_title(f'cGAN reconstructed sharpness= {ash}')

axs[2].imshow(miniplot3z46, cmap='gray',vmin=vmin,vmax=vmax,interpolation='none', extent=[90,120,32,0])
axs[2].axis('off')
axs[2].set_title(f'Subsampled sharpeness= {csh}')

plt.subplots_adjust(wspace=0.05, hspace=0)
figname = f'comparision edges z={z}_xint{xint}_xfin{xfin}_yint{yint}_yfin{yfin}.png'
cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[2], label='dB')  
plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
plt.show()
print(sharpness(a),sharpness(b),sharpness(c))

#%%
x = 520
plot1 = 10*np.log10(abs(tomReconstructed[:,x,:,0]+1j*tomReconstructed[:,x,:,1])**2)
plot2 = 10*np.log10(abs(tomOriginal[:,x,:,0]+1j*tomOriginal[:,x,:,1])**2)
# plot3 = 10*np.log10(abs(tomSubsampled[:,x,:,0]+1j*tomSubsampled[:,x,:,1])**2)
plot3 = 10*np.log10(subsampled2[:,x,:])
# plt.imshow(plot3, cmap='gray',vmin=60,vmax=120)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].imshow(plot2, cmap='gray',vmin=60,vmax=120,aspect='equal')
axs[0].axis('off')
axs[0].set_title('Original')

axs[1].imshow(plot1, cmap='gray',vmin=60,vmax=120,aspect='equal')
axs[1].axis('off') 
axs[1].set_title('cGAN reconstructed')

axs[2].imshow(plot3, cmap='gray',vmin=60,vmax=120,aspect='equal')
axs[2].axis('off')
axs[2].set_title('Subsampled')

plt.subplots_adjust(wspace=0.05, hspace=0)
figname = f'comparision x={x}.png'
# plt.savefig(os.path.join(path,figname), dpi=300)
plt.show()
#%%
y = 414
plot1 = 10*np.log10(abs(tomReconstructed[:,:,y,0]+1j*tomReconstructed[:,:,y,1])**2)
plot2 = 10*np.log10(abs(tomOriginal[:,:,y,0]+1j*tomOriginal[:,:,y,1])**2)
plot3 = 10*np.log10(abs(tomSubsampled[:,:,int(y/2),0]+1j*tomSubsampled[:,:,int(y/2),1])**2)
# plt.imshow(plot3, cmap='gray',vmin=60,vmax=120)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(plot2, cmap='gray',vmin=60,vmax=120,aspect='equal')
axs[0].axis('off')
axs[0].set_title('Original')

axs[1].imshow(plot1, cmap='gray',vmin=60,vmax=120,aspect='equal')
axs[1].axis('off') 
axs[1].set_title('cGAN reconstructed')

axs[2].imshow(plot3, cmap='gray',vmin=60,vmax=120,aspect='equal')
axs[2].axis('off')
axs[2].set_title('Subsampled')

plt.subplots_adjust(wspace=0.05, hspace=0)
figname = f'comparision y={y}.png'
plt.savefig(os.path.join(path,figname), dpi=300)
plt.show()
#%% save hd image
# fig, ax = plt.subplots()
# ax.imshow(plot3, cmap='gray',vmin=85,vmax=110)  # Puedes cambiar 'viridis' por el colormap que prefieras.
# # Elimina los ejes y bordes blancos
# ax.axis('off')
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# # Guarda la imagen
# plt.savefig("ZX_Fovea_subsampled.png", bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()
#%%
z = 170
enfaceReconstructed = tomReconstructed[z,:,:,:]
correlationx,correlationy = Correlation(enfaceReconstructed)
stdxr = np.std(correlationx)
meanxr = np.mean(correlationx)
stdyr = np.std(correlationy)
meanyr = np.mean(correlationy)
plt.imshow(correlationy,cmap='twilight')
#%%
fig, ax = plt.subplots()
ax.imshow(correlationx, cmap='twilight')  # Puedes cambiar 'viridis' por el colormap que prefieras.
# Elimina los ejes y bordes blancos
ax.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# Guarda la imagen
plt.savefig("correlationx_reconstructed.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
#%%
enfaceOriginal = tomOriginal[z,:,:,:]
correlationx,correlationy = Correlation(enfaceOriginal)
stdxo = np.std(correlationx)
meanxo = np.mean(correlationx)
stdyo = np.std(correlationy)
meanyo = np.mean(correlationy)
plt.imshow(correlationy,cmap='twilight')
#%%
fig, ax = plt.subplots()
ax.imshow(correlationx, cmap='twilight')  # Puedes cambiar 'viridis' por el colormap que prefieras.
# Elimina los ejes y bordes blancos
ax.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# Guarda la imagen
plt.savefig("correlationx_original.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
#%%

enfaceSubsampled = np.stack((np.real(subsampled2[z,:,:]),
                            np.imag(subsampled2[z,:,:])),axis=2)
correlationx,correlationy = Correlation(enfaceSubsampled)
stdxs = np.std(correlationx)
meanxs = np.mean(correlationx)
stdys = np.std(correlationy)
meanys = np.mean(correlationy)
plt.imshow(correlationy,cmap='twilight')
#%%
fig, ax = plt.subplots()
ax.imshow(correlationy, cmap='twilight')  # Puedes cambiar 'viridis' por el colormap que prefieras.
# Elimina los ejes y bordes blancos
ax.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# Guarda la imagen
plt.savefig("correlationy_subsampled.png", bbox_inches='tight', pad_inches=0, dpi=72)
plt.close()
#%%
mpsReconstructedx = MPS_single(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1],meandim=0)
mpsReconstructedy = MPS_single(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1],meandim=1)
plt.plot(mpsReconstructedy)
#%%
mpsOriginalx = MPS_single(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1],meandim=0)
mpsOriginaly = MPS_single(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1],meandim=1)
plt.plot(mpsOriginalx)
plt.show()

#%%
mpsSubsampledx = MPS_single(tomSubsampled[z,:,:,0]+1j*tomSubsampled[z,:,:,1],meandim=0)
mpsSubsampledy = MPS_single(tomSubsampled[z,:,:,0]+1j*tomSubsampled[z,:,:,1],meandim=1)
plt.plot(mpsSubsampledy)
plt.show()
#%%
from skimage.metrics import structural_similarity as ssim
import numpy as np
def calculate_ssim(image1, image2):
    """
    Calcula el Índice de Similitud Estructural (SSIM) entre dos imágenes.

    Parámetros:
    - image1, image2: numpy arrays 2D representando las imágenes a comparar.

    Devuelve:
    - SSIM entre las dos imágenes.
    """
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño."
    
    # El SSIM se calcula típicamente en imágenes de 8 bits (0-255)
    ssim_value, _ = ssim(image1, image2, full=True,data_range=1)
    return ssim_value

def calculate_mse(image_original, image_reconstructed):
    """
    Calcula el Error Cuadrático Medio (MSE) entre dos imágenes.

    Parámetros:
    - image_original: numpy array 2D representando la imagen original en escala de grises.
    - image_reconstructed: numpy array 2D representando la imagen reconstruida por la cGAN.

    Devuelve:
    - MSE entre las dos imágenes.
    """
    assert image_original.shape == image_reconstructed.shape, "Las imágenes deben tener el mismo tamaño."

    mse = np.mean((image_original - image_reconstructed) ** 2)
    return mse

def calculate_psnr(image1, image2, max_val=1.0):
    """
    Calcula la Relación Señal-Ruido de Pico (PSNR) entre dos imágenes.

    Parámetros:
    - image1, image2: numpy arrays 2D representando las imágenes a comparar.
    - max_val: Valor máximo posible de la señal de la imagen. Por defecto es 1.0, para imágenes normalizadas.

    Devuelve:
    - PSNR entre las dos imágenes.
    """
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño."
    
    # Calcular el Error Cuadrático Medio (MSE)
    mse = np.mean((image1 - image2) ** 2)
    
    # Evitar un MSE de 0 (que daría un PSNR infinito)
    if mse == 0:
        return float('inf')

    # Calcular PSNR
    psnr = 10 * np.log10(max_val**2 / mse)
    
    return psnr

def relative_error(true_image, estimated_image):
    # Calculamos el error relativo para cada píxel
    error = np.abs(true_image - estimated_image) / (true_image + 1e-10)  # Evitamos división por cero
    # Promediamos el error relativo en toda la imagen
    mean_relative_error = np.mean(error)
    return mean_relative_error

import cv2

def histogram_difference(image1, image2, method="chi-squared"):
    # Calcular histogramas
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 1]) # asumiendo que la imagen está normalizada entre 0 y 1
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 1])

    # Normalizar histogramas si se va a usar Kullback-Leibler
    if method == "kullback-leibler":
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
    
    # Calcular diferencia de histogramas
    if method == "chi-squared":
        # Usar distancia chi-cuadrado
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    elif method == "kullback-leibler":
        # Usar divergencia de Kullback-Leibler
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
    else:
        raise ValueError("Método no reconocido")


enfaceOriginal=10*np.log10(abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2)
enfaceOriginal = (enfaceOriginal/np.max(enfaceOriginal))
enfaceReconstructed =10*np.log10(abs(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1])**2)
enfaceReconstructed = (enfaceReconstructed/np.max(enfaceReconstructed))
ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
rerror = relative_error(enfaceOriginal,enfaceReconstructed)
hdiff = histogram_difference(np.float32(enfaceOriginal),np.float32(enfaceReconstructed))
print("SSIM:", ssim_result)
print("MSE:", mse)
print("PSNR:", psnr)
print("relative error:", rerror)
print("histogram difference:", hdiff)

#%%
px.imshow(plot3,color_continuous_scale='gray',zmin=85,zmax=110)

#%%

tomSubsampled = np.pad(tomSubsampled, ((0, 0), (0, 0), (128, 128) ,(0,0)), mode='constant', constant_values=1)

#%%
tomint = np.array([abs(tomOriginal[:,:,:,0]+1j*tomOriginal[:,:,:,1]),
          abs(tomReconstructed[:,:,:,0]+1j*tomReconstructed[:,:,:,1]),
          abs(tomSubsampled[:,:,:,0]+1j*tomSubsampled[:,:,:,1])])
#%%
original_array = np.transpose(tomint, (1, 2, 3, 0))
# Redimensionamos los volúmenes para que tengan tres dimensiones (x, y, 2z o 3z)
volume1 = original_array[:,:,:,0].reshape((1024, 1152, -1))
volume2 = original_array[:,:,:,1].reshape((1024, 1152, -1))
volume3 = original_array[:,:,:,2].reshape((1024, 1152, -1))  # Extrae y redimensiona tomSub

# Concatenamos los volúmenes a lo largo del eje Y
compare = np.concatenate((volume1, volume2, volume3), axis=2)
# compare = np.transpose(compare, (2, 0, 1))
del volume1, volume2, volume3

#%%
def logint(data):
    return 10*np.log10(abs(data)**2)

z = 170
bscanOriginal = 10*np.log10(abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2)
bscanSubsmapled = logint(subsampled2[z,:,:])
bscancgan = logint(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1])
#%%
dif2 = bscanOriginal - bscancgan
bscanOriginal = abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2
bscanSubsmapled = abs(subsampled2)**2
bscancgan = abs(tomReconstructed[z,:,:,0]+1j*tomReconstructed[z,:,:,1])**2
diflog = 10*np.log10(bscanOriginal-bscanSubsmapled)
#%%
from numpy.fft import fft,fftshift,ifftshift
def myfft(data):
    return (abs(ifftshift(fft(fftshift(data)))))

aa = myfft(np.pad(miniplot2z46[:,299],(0,1200)))
bb = myfft(np.pad(miniplot1z46[:,299],(0,1200)))
cc = myfft(np.pad(miniplot3z46[:,299],(0,1200)))
plt.plot(10*np.log10(aa),label='original')
plt.plot(10*np.log10(bb),label='cgan')
plt.plot(10*np.log10(cc),label='interpolado')
plt.legend()
# %%

def mypad(data,length):
    return np.pad(data,(0,length))
meanoriginal = myfft(mypad(np.mean(miniplot2z46,axis=0),1200))**2
meancgan = myfft(mypad(np.mean(miniplot1z46,axis=0),1200))**2
meansub = myfft(mypad(np.mean(miniplot3z46,axis=0),1200))**2
plt.plot(meanoriginal[300:600],label='original')
plt.plot(meancgan[300:600],label='cgan')
plt.plot(meansub[300:600],label='interpolado')
plt.legend()
plt.show()
