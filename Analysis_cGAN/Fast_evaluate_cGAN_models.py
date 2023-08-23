#%% import libraries
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Deep_Utils import simple_sliding_window
from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Metrics import ownPhaseMetricCorrected_numpy, ssimMetric, ownPhaseMetric_numpy
import os
import gc
from tensorflow.keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#%%

root=r'C:\Users\USER\Documents\cGAN_1'
model_folder = '\\Models'
path = root+model_folder
savefolder = path
""" Load tomograms"""
pathorig = r'C:\Users\USER\Documents\GitHub\Fovea'
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
#%%
q=3
zini = 170
zfin = zini + q
tomDatas= tomDatas[zini:zfin,:,:,:]
num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros), (0, 0))
tomDatas = np.pad(tomDatas, pad_width, mode='edge')
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
for i in models:
    print('__________ reading___________',i)
    model = tf.keras.models.load_model(path+'\\'+ i,compile=False)
    print('model loaded')
    logslicesOver = np.array(model.predict(logslicesUnder, batch_size=4), dtype='float64')
    slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
    slicesUnder=downSampleSlices(slices)
    print('metrics evaluation')
    ssims = np.mean(ssimMetric(logslices, logslicesOver))
    phasemetric = np.mean(ownPhaseMetric_numpy(logslices, logslicesOver))
    phasemetricCorrected = np.mean(ownPhaseMetricCorrected_numpy(logslices, logslicesOver))
    mse = np.mean((logslices - logslicesOver)**2)
    epoch_metrics = np.array((ssims,phasemetric,phasemetricCorrected,mse))
    metrics.append(epoch_metrics)
    del model
    gc.collect()
    K.clear_session()
metrics_log = np.array(metrics)
np.save(root+'\\metrics_log',metrics_log)
#%%
ssims = metrics_log[:,0]
phasemetric = metrics_log[:,1]
phasemetricCorrected = metrics_log[:,2]
mse = metrics_log[:,3]
fig,axs = plt.subplots(2,2)
axs[0,0].plot(ssims)
axs[0,0].set_title('ssims')
axs[0,1].plot(mse)
axs[0,1].set_title('mse')
axs[1,0].plot(phasemetricCorrected)
axs[1,0].set_title('phase corrected')
axs[1,1].plot(phasemetric)
axs[1,1].set_title('phase')

#%%
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

