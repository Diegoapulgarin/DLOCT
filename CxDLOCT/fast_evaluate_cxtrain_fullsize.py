#%%
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
from Metrics import ownPhaseMetricCorrected_numpy, ssimMetric, ownPhaseMetric_numpy
from tensorflow.keras import backend as K
import gc
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Activation

def dbscale(darray):
    if len(np.shape(darray))==3:
        img = 10*np.log10(abs(darray[:,:,0]+1j*darray[:,:,1])**2)
    else:
        img = 10*np.log10(abs(darray[:,:])**2)
    return img

def logScale(slices):
    
    logslices = np.copy(slices)
    nSlices = slices.shape[2]
    logslicesAmp = abs(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # and retrieve the phase
    logslicesPhase = np.angle(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # reescale amplitude
    logslicesAmp = np.log10(logslicesAmp)
    slicesMax = np.reshape(logslicesAmp.max(axis=(0, 1)), ( 1, 1,nSlices))
    slicesMin = np.reshape(logslicesAmp.min(axis=(0, 1)), ( 1, 1,nSlices))
    logslicesAmp = (logslicesAmp - slicesMin) / (slicesMax - slicesMin)
    # --- here, we could even normalize each slice to 0-1, keeping the original
    # --- limits to rescale after the network processes
    # and redefine the real and imaginary components with the new amplitude and
    # same phase
    logslices[:, :, :, 0] = (np.real(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    logslices[:, :, :, 1] = (np.imag(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    
    return logslices, slicesMax, slicesMin

def inverseLogScale(oldslices, slicesMax, slicesMin):
 
    slices = np.copy(oldslices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesPhase = np.angle(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesAmp = slicesAmp * (slicesMax - slicesMin) + slicesMin
    slicesAmp = 10**(slicesAmp)
    slices[:, :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices

def inverseLogScaleSummary(oldslices, slicesMax, slicesMin):
 
    slices = np.copy(oldslices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[ :, :, 0] + 1j*slices[ :, :, 1])
    slicesPhase = np.angle(slices[ :, :, 0] + 1j*slices[ :, :, 1])
    slicesAmp = slicesAmp * (slicesMax - slicesMin) + slicesMin
    slicesAmp = 10**(slicesAmp)
    slices[:, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices

def extract_dimensions(file_name):
    parts = file_name.split('_')
    dimensions = []
    for part in parts:
        if 'z=' in part or 'x=' in part or 'y=' in part:
            number = int(part.split('=')[-1])
            dimensions.append(number)
    return tuple(dimensions)

def read_tomogram(file_path, dimensions):
    depth, height, width = dimensions
    with open(file_path, 'rb') as file:
        tomogram = np.fromfile(file, dtype='single')
        tomogram = tomogram.reshape((depth, height, width),order='F')
    return tomogram

def max_value(array):
    max_val = array[0]  # initialize maximum value as the first element of the array
    max_pos = 0         # initialize maximum position as 0
    for i in range(1, len(array)):
        if array[i] > max_val:
            max_val = array[i]  # update maximum value
            max_pos = i         # update maximum position
    return max_val, max_pos

def min_value(array):
    min_val = array[0]  # initialize maximum value as the first element of the array
    min_pos = 0         # initialize maximum position as 0
    for i in range(1, len(array)):
        if array[i] < min_val:
            min_val = array[i]  # update maximum value
            min_pos = i         # update maximum position
    return min_val, min_pos

def custom_gelu(x):
    return gelu(x, approximate=True)

#%%
dataPath = r'E:\DLOCT\Experimental_Data_complex'
tom = np.load(os.path.join(dataPath,'validationOpticNerve.npy'))
tomcc = np.load(os.path.join(dataPath,'validationOpticNervecc.npy'))
# tom = np.load(os.path.join(dataPath,'depthNail.npy'))
# tomcc = np.load(os.path.join(dataPath,'depthNailcc.npy'))

tomNorm,tmax,tmin = logScale(tom)
tomNorm = np.transpose(tomNorm, (2, 0, 1, 3))
tomccNorm,imax,imin = logScale(tomcc)
tomccNorm = np.transpose(tomccNorm, (2, 0, 1, 3))
print('tomograms loaded')

#%%
modelsPath = r'E:\models\cxpix2pixcomplexdbscale4\models'
listmodels = os.listdir(modelsPath)
metrics = []
metricsS = []
for model in tqdm(listmodels):
    model_loaded = tf.keras.models.load_model(os.path.join(modelsPath,model), 
                                              compile=False)
    tomPredict = np.array(model_loaded.predict(tomccNorm, batch_size=1), dtype='float32')
    # tomPredictS = np.array(model_loaded.predict(tomccNormS, batch_size=1), dtype='float32')
    ssims = np.mean(ssimMetric(tomNorm, tomPredict))
    ssims_std = np.std(ssimMetric(tomNorm, tomPredict))
    ssims_uncertainty = ssims_std / np.sqrt(len(tomPredict))
    mse = np.mean((tomNorm - tomPredict)**2)
    mse_std = np.std((tomNorm - tomPredict)**2)
    mse_uncertainty = mse_std / np.sqrt(np.prod(np.shape(mse)))
    epoch_metrics = np.array((ssims,ssims_std,ssims_uncertainty,mse,mse_std,mse_uncertainty))
    metrics.append(epoch_metrics)
    del model_loaded
    gc.collect()
    K.clear_session()
metrics_log = np.array(metrics)
metricsS_log = np.array(metricsS)

ssims = metrics_log[:,0]
mse = metrics_log[:,3]
fig,axs = plt.subplots(1,2,figsize=(15,5))
axs[0].plot(ssims[1:])
axs[0].set_title('ssims')
axs[1].plot(mse[1:])
axs[1].set_title('mse')
fig.suptitle('Metrics for experimentals', fontsize=16)


#%%
print('___________Experimentals_____________')
max_val, max_pos = max_value(ssims)
print(f"The maximum value of the ssim metric is {np.round(max_val,2)}, and the model is {listmodels[max_pos]}.")

min_val, min_pos = min_value(ssims)
print(f"The minimum value of the ssim metric is {np.round(min_val,2)} and the model is {listmodels[min_pos]}.")

max_val, max_pos = max_value(mse)
print(f"The maximum value of the mse metric is {np.round(max_val,2)} and the model is {listmodels[max_pos]}.")

min_val, min_pos = min_value(mse)
print(f"The minimum value of the mse metric is {np.round(min_val,2)} and the model is {listmodels[min_pos]}.")
#%%
model = listmodels[max_pos]
model_loaded = tf.keras.models.load_model(os.path.join(modelsPath,model), 
                                            compile=False)
tomPredict = np.array(model_loaded.predict(tomccNorm, batch_size=1), dtype='float32')
gc.collect()
K.clear_session()
tomPredictreordered = np.transpose(tomPredict, (1, 2, 0, 3))
tomPredictFullScale = inverseLogScale(tomPredictreordered,imax,imin)
#%%
vmax = 100
vmin = 50
bscan = 0
fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(20,15))
axs[0].imshow(dbscale(tomcc[:,:,bscan,:]),cmap='gray',vmax=vmax,vmin=vmin)
axs[0].axis('off')
axs[0].set_title('Input')
axs[1].imshow(dbscale(tomPredictFullScale[:,:,bscan,:]),cmap='gray',vmax=vmax,vmin=vmin)
axs[1].axis('off')
axs[1].set_title('Neural Network')
axs[2].imshow(dbscale(tom[:,:,bscan,:]),cmap='gray',vmax=vmax,vmin=vmin)
axs[2].axis('off')
axs[2].set_title('Target')