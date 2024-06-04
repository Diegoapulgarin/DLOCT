#%% import libraries
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Deep_Utils import simple_sliding_window,simple_inv_sliding_window,dbscale,Correlation
from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices, downSampleSlicesInterp
from Metrics import ownPhaseMetricCorrected_numpy, ssimMetric, ownPhaseMetric_numpy
import os
import gc
import plotly.express as px
from tensorflow.keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#%%

root=r'E:\DLOCT\ultimo experimento subsampling paper\Fovea\cGAN_exp'
model_folder = '\\Models'
path = root+model_folder
savefolder = path
""" Load tomograms"""
pathorig = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
tomreal = np.fromfile(pathorig+'\\'+filename+real,'single')
tomreal = np.reshape(tomreal,(586,896,960,2),order='F')
tomreal = np.sum(tomreal,axis=3)
tomimag = np.fromfile(pathorig+'\\'+filename+imag,'single')
tomimag = np.reshape(tomimag,(586,896,960,2),order='F')
tomimag = np.sum(tomimag,axis=3)
tomDatas = np.stack((tomreal,tomimag), axis=3)
del tomimag, tomreal
print('Tomogram loaded')
#%%
q=3
zini = 170
zfin = zini + q
tomDatas= tomDatas[zini:zfin,:,:,:]
num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros), (0, 0))
tomDatas = np.pad(tomDatas, pad_width, mode='reflect')
print(np.shape(tomDatas))
#%%
slidingYSize = 128
slidingXSize = 128
strideY = 128
strideX = 128
tomShape = np.shape(tomDatas)
slices = simple_sliding_window(tomDatas,tomShape,slidingYSize,slidingXSize,strideY,strideX)
print(np.shape(slices))
logslices, slicesMax, slicesMin = logScaleSlices(slices)
logslicesUnder = downSampleSlices(logslices)

#%%
models = os.listdir(path)
metrics = []
metricsFullsize = []
for i in models:
    print('__________ reading___________',i)
    model = tf.keras.models.load_model(path+'\\'+ i,compile=False)
    print('model loaded')
    logslicesOver = np.array(model.predict(logslicesUnder, batch_size=8), dtype='float64')
    # slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
    # tomOver = simple_inv_sliding_window(slicesOver,
    #                                     tomShape,
    #                                     slidingYSize,
    #                                     slidingXSize,
    #                                     strideY,
    #                                     strideX)
    print('metrics evaluation')
    ssims = np.mean(ssimMetric(logslices, logslicesOver))
    ssims_std = np.std(ssimMetric(logslices, logslicesOver))
    ssims_uncertainty = ssims_std / np.sqrt(len(logslices))
    phasemetric = np.mean(ownPhaseMetric_numpy(logslices, logslicesOver))
    phasemetricCorrected = np.mean(ownPhaseMetricCorrected_numpy(logslices, logslicesOver))
    mse = np.mean((logslices - logslicesOver)**2)
    mse_std = np.std((logslices - logslicesOver)**2)
    mse_uncertainty = mse_std / np.sqrt(np.prod(np.shape(mse)))
    epoch_metrics = np.array((ssims,ssims_std,ssims_uncertainty,phasemetric,phasemetricCorrected,mse,mse_std,mse_uncertainty))
    metrics.append(epoch_metrics)

    # ssims = np.mean(ssimMetric(tomDatas, tomOver))
    # ssims_std = np.std(ssimMetric(tomDatas, tomOver))
    # ssims_uncertainty = ssims_std / np.sqrt(len(tomDatas))
    # phasemetric = np.mean(ownPhaseMetric_numpy(tomDatas, tomOver))
    # phasemetricCorrected = np.mean(ownPhaseMetricCorrected_numpy(tomDatas, tomOver))
    # mse = np.mean((tomDatas - tomOver)**2)
    # mse_std = np.std((tomDatas - tomOver)**2)
    # mse_uncertainty = mse_std / np.sqrt(np.prod(np.shape(mse)))
    # epoch_metrics = np.array((ssims,ssims_std,ssims_uncertainty,phasemetric,phasemetricCorrected,mse,mse_std,mse_uncertainty))
    # metricsFullsize.append(epoch_metrics)
    del model
    gc.collect()
    K.clear_session()
#%%
metrics_log = np.array(metrics)
np.save(root+'\\metrics_log',metrics_log)
ssims = metrics_log[:,0]
phasemetric = metrics_log[:,1]
phasemetricCorrected = metrics_log[:,2]
mse = metrics_log[:,6]
fig,axs = plt.subplots(2,2)
axs[0,0].plot(ssims)
axs[0,0].set_title('ssims')
axs[0,1].plot(mse)
axs[0,1].set_title('mse')
axs[1,0].plot(phasemetricCorrected)
axs[1,0].set_title('phase corrected')
axs[1,1].plot(phasemetric)
axs[1,1].set_title('phase')

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

max_val, max_pos = max_value(ssims)
print(f"The maximum value of the ssim metric is {max_val} and its position is {max_pos}.")

min_val, min_pos = min_value(ssims)
print(f"The minimum value of the ssim metric is {min_val} and its position is {min_pos}.")

max_val, max_pos = max_value(mse)
print(f"The maximum value of the mse metric is {max_val} and its position is {max_pos}.")

min_val, min_pos = min_value(mse)
print(f"The minimum value of the mse metric is {min_val} and its position is {min_pos}.")

max_val, max_pos = max_value(phasemetric)
print(f"The maximum value of the phase metric is {max_val} and its position is {max_pos}.")

min_val, min_pos = min_value(phasemetric)
print(f"The minimum value of the phase metric is {min_val} and its position is {min_pos}.")

max_val, max_pos = max_value(phasemetricCorrected)
print(f"The maximum value of the phase metric corrected is {max_val} and its position is {max_pos}.")

min_val, min_pos = min_value(phasemetricCorrected)
print(f"The minimum value of the phase metric corrected is {min_val} and its position is {min_pos}.")
#%%
# max_val, max_pos = max_value(ssims)
# best = max_pos
best = 90
model = tf.keras.models.load_model(path+'\\'+ models[best],compile=False)
logslicesOver = np.array(model.predict(logslicesUnder, batch_size=8), dtype='float64')
slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
tomOver = simple_inv_sliding_window(slicesOver,tomShape,slidingYSize,slidingXSize,strideY,strideX)
tomOver = tomOver[:,:,0:960,:]
tomDatas = tomDatas[:,:,0:960,:]

z= 0
tom = dbscale(tomOver[z,:,:,:])
tomOriginal = dbscale(tomDatas[z,:,:,:])
plt.imshow(tom,cmap='gray',vmax=120,vmin=70)
plt.title('cGAN')
plt.xticks([])  
plt.yticks([])  
plt.show()
plt.imshow(tomOriginal,cmap='gray',vmax=120,vmin=70)
plt.title('Original')
plt.xticks([])  
plt.yticks([])  
plt.show()
#%%



enfaceReconstructed = tomOver[z,:,:,:]
correlationReconstructedx,correlationReconstructedy = Correlation(enfaceReconstructed)
stdxr = np.std(correlationReconstructedx)
meanxr = np.mean(correlationReconstructedx)
stdyr = np.std(correlationReconstructedy)
meanyr = np.mean(correlationReconstructedy)
filenamex = f'correlationx_Reconstructed_z={z}_mean={meanxr}_std={stdxr}.png'
filenamey = f'correlationy_Reconstructed_z={z}_mean={meanyr}_std={stdyr}.png'
plt.imshow(correlationReconstructedy,cmap='twilight')
plt.title('Reconstructed correlation')
plt.xticks([])  
plt.yticks([])  
plt.show()

enfaceOriginal = tomDatas[z,:,:,:]
correlationOriginalx,correlationOriginaly = Correlation(enfaceOriginal)
stdxo = np.std(correlationOriginalx)
meanxo = np.mean(correlationOriginalx)
stdyo = np.std(correlationOriginaly)
meanyo = np.mean(correlationOriginaly)
filenamex = f'correlationx_original_z={z}_mean={meanxo}_std={stdxo}.png'
filenamey = f'correlationy_original_z={z}_mean={meanyo}_std={stdyo}.png'
plt.imshow(correlationOriginaly,cmap='twilight')
plt.title('Original correlation')
plt.xticks([])  
plt.yticks([])  
plt.show()
# %%
