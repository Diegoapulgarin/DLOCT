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

#%%
CentralWavelength = 870e-9
bandwith = 50e-9
pixel = (2*np.log(2)/np.pi)*(CentralWavelength**2/bandwith)

#%%
path = r'C:\Users\USER\Documents\GitHub\Fovea'
#%%
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 586
nXbin = 896
nYbin = 960
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,2),order='F')
tom = np.sum(tom,axis=3)

tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,2),order='F')
tomi = np.sum(tomi,axis=3)

tomOriginal = np.stack((tom, tomi), axis=3)
del tom, tomi
#%%
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 586
nXbin = 896
nYbin = 960
tom = np.fromfile(path+'\\'+filename+real,'single')
tom = np.reshape(tom,(nZbin,nXbin,nYbin),order='F')

tomi = np.fromfile(path+'\\'+filename+imag,'single')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin),order='F')


tomReconstructed = np.stack((tom, tomi), axis=3)
del tom, tomi

#%%
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..512)_y=(1..960)_subsampled'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 586
nXbin = 448
nYbin = 960
tom = np.fromfile(path+'\\'+filename+real,'single')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,2),order='F')
tom = np.sum(tom,axis=3)

tomi = np.fromfile(path+'\\'+filename+imag,'single')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,2),order='F')
tomi = np.sum(tomi,axis=3)


tomSubsampled = np.stack((tom, tomi), axis=3)
del tom, tomi
#%%
z = 170
plot1 = 10*np.log10(abs(tomReconstructed[z,:,:])**2)
plot2 = 10*np.log10(abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2)
#%%
z =170
plot3 = 10*np.log10(abs(tomSubsampled[z,:,:,0]+1j*tomSubsampled[z,:,:,1])**2)
plt.imshow(plot3, cmap='gray',vmin=85,vmax=110)
#%%

fig, ax = plt.subplots()
ax.imshow(plot3, cmap='gray',vmin=85,vmax=110)  # Puedes cambiar 'viridis' por el colormap que prefieras.
# Elimina los ejes y bordes blancos
ax.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

# Guarda la imagen
plt.savefig("ZX_Fovea_subsampled.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()

#%%
z = 170
enfaceReconstructedReal = np.real(tomReconstructed[z,:,:])
enfaceReconstructedImag=np.imag(tomReconstructed[z,:,:])
enfaceReconstructed = np.stack((enfaceReconstructedReal,enfaceReconstructedImag),axis=2)
correlationx,correlationy = Correlation(enfaceReconstructed)
stdxr = np.std(correlationx)
meanxr = np.mean(correlationx)
stdyr = np.std(correlationy)
meanyr = np.mean(correlationy)
#%%
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
z = 170
enfaceOriginal = tomOriginal[z,:,:,:]
correlationx,correlationy = Correlation(enfaceOriginal)
stdxo = np.std(correlationx)
meanxo = np.mean(correlationx)
stdyo = np.std(correlationy)
meanyo = np.mean(correlationy)
#%%
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
z = 170
enfaceSubsampled = tomSubsampled[z,:,:,:]
correlationx,correlationy = Correlation(enfaceSubsampled)
stdxs = np.std(correlationx)
meanxs = np.mean(correlationx)
stdys = np.std(correlationy)
meanys = np.mean(correlationy)
print(stdxs)
print(stdys)

#%%
plt.imshow(correlationx,cmap='twilight')
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
z=170
mpsReconstructedx = MPS_single(tomReconstructed[z,:,:],meandim=0)
mpsReconstructedy = MPS_single(tomReconstructed[z,:,:],meandim=1)
mpsOriginalx = MPS_single(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1],meandim=0)
mpsOriginaly = MPS_single(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1],meandim=1)
#%%
plt.plot(mpsReconstructedy)
# plt.plot(mpsOriginaly)
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
    # Asegúrate de que las imágenes tengan el mismo tamaño y tipo de dato
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño."
    
    # El SSIM se calcula típicamente en imágenes de 8 bits (0-255)
    # Si tus imágenes no están en este rango, podrías considerar normalizarlas o adaptar el rango
    # Por simplicidad, aquí supondré que están en el rango [0, 1] (por ejemplo, tras una normalización)
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
    # Asegúrate de que las imágenes tengan el mismo tamaño
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
    # Asegúrate de que las imágenes tengan el mismo tamaño y tipo de dato
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



#%%
z = 170
enfaceOriginal=10*np.log10(abs(tomOriginal[z,:,:,0]+1j*tomOriginal[z,:,:,1])**2)
enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
enfaceReconstructed =10*np.log10(abs(tomReconstructed[z,:,:])**2)
enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)
ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
rerror = relative_error(enfaceOriginal,enfaceReconstructed)
hdiff = histogram_difference(enfaceOriginal,enfaceReconstructed)
print("SSIM:", ssim_result)
print("MSE:", mse)
print("PSNR:", psnr)
print("relative error:", rerror)
print("histogram difference:", hdiff)

#%%
px.imshow(plot3,color_continuous_scale='gray',zmin=85,zmax=110)