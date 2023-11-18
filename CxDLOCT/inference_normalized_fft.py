#%% libraries
import numpy as np
from numpy.random import randn
import tensorflow as tf
from scipy.fft import fft, fftshift
import scipy.io as sio
import matplotlib.pyplot as plt
#%% functions
def normalize_aline(aline):
    min_val = np.min(aline)
    range_val = np.max(aline) - min_val
    normalized_aline = (aline - min_val) / range_val
    return normalized_aline, min_val, range_val

def normalize_volume_by_aline(volume):
    z, x, y = volume.shape
    normalized_volume = np.zeros_like(volume)
    min_vals = np.zeros((x, y))
    range_vals = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            normalized_volume[:, i, j], min_vals[i, j], range_vals[i, j] = normalize_aline(volume[:, i, j])
            
    return normalized_volume, min_vals, range_vals

def inverse_normalize_aline(normalized_aline, min_val, range_val):
    original_aline = normalized_aline * range_val + min_val
    return original_aline

def inverse_normalize_volume_by_aline(normalized_volume, min_vals, range_vals):
    z, x, y = normalized_volume.shape
    original_volume = np.zeros_like(normalized_volume)
    
    for i in range(x):
        for j in range(y):
            original_volume[:, i, j] = inverse_normalize_aline(normalized_volume[:, i, j], min_vals[i, j], range_vals[i, j])
            
    return original_volume

def reconstruct_tomogram(fringes1, zeroPadding=0, noiseFloorDb=0,z=2):
    nK = fringes1.shape[0]  # the size along the first dimension
    nZ, nX, nY = fringes1.shape  # fringes1 is 3D
    zRef = nZ / z  # zRef value
    zSize = 256  # zSize value

    # Apply hanning window along the first dimension
    fringes1 = fringes1 * np.hanning(nK)[:, np.newaxis, np.newaxis]

    # Pad the fringes
    fringes1_padded = np.pad(fringes1, ((zeroPadding, zeroPadding), (0, 0), (0, 0)), mode='constant')

    # Fourier Transform
    tom1True = fftshift(fft(fftshift(fringes1_padded, axes=0), axis=0), axes=0)
    tom1 = tom1True + (((10 ** (noiseFloorDb / 20)) / 1) * (randn(nZ, nX, nY) + 1j * randn(nZ, nX, nY)))

    refShift = int((2 * zRef + zSize) / zSize * nZ) // 2
    tom1 = np.roll(tom1, refShift, axis=0)
    tom1True = np.roll(tom1True, refShift, axis=0)
    
    return tom1True, tom1

def consistent_zero_padding(volume, target_z_size, start_fraction=0.5):
    z, x, y = volume.shape
    
    if z >= target_z_size:
        return volume[:target_z_size, :, :]
    
    padding_size = target_z_size - z
    start_padding = int(padding_size * start_fraction)
    end_padding = padding_size - start_padding
    
    padded_volume = np.pad(volume, ((start_padding, end_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    return padded_volume
#%% 
path = r'C:\Users\USER\Documents\GitHub\Simulated_Data_Complex'
filename = 'Spheresv2_256x256x16_7_sdarr.mat'
mat_contents = sio.loadmat(path+'/'+filename)
fringes1 = mat_contents['fringes1']
#%%
normalized_volume_real = np.zeros(np.shape(fringes1))
min_vals_list_real =[]
range_vals_list_real =[]

fftfringes,_ = reconstruct_tomogram(np.real(fringes1[:,:,:]),z=2)
normalized_volume_real[:,:,:], min_vals, range_vals = normalize_volume_by_aline(abs(fftfringes))
min_vals_list_real.append(min_vals)
range_vals_list_real.append(range_vals)
#%% plot fft the first aline
plt.figure()
plt.plot(fftfringes[:, 1, 1], label="Real signal fft")
plt.legend()
plt.show()
#%% plot bscan 
plt.imshow(abs(fftfringes[:, :, 1]))
#%%
padded_tomogram = consistent_zero_padding(normalized_volume_real,target_z_size=1024,start_fraction=0.5)
#%%
plt.imshow(abs(padded_tomogram[:, :, 1]))
#%%
padded_tomogram_predict = np.transpose(padded_tomogram,(1,2,0))
x, y, z = padded_tomogram_predict.shape
X = np.reshape(padded_tomogram_predict,(x*y,z))
plt.figure()
plt.plot(X[1,:], label="Real signal fft")
plt.legend()
plt.show()
#%%
path_model = r'C:\Users\USER\Documents\GitHub\models cxDLOCT\first_run'
model = tf.keras.models.load_model(path_model+'\\model_049152.h5')
#%%
Y = model.predict(X)
#%%
predicted_volume=np.reshape(Y,(x,y,z))
predicted_volume = np.transpose(predicted_volume,((2,0,1)))
predicted_volume = predicted_volume[384:384+256,:,:]

# plt.figure()
# plt.imshow(predicted_volume[:,:,0], label="complex signal fft")
# plt.legend()
# plt.show()
#%%
normalized_volume_target = np.zeros(np.shape(fringes1))
min_vals_list_target =[]
range_vals_list_target =[]

fftfringes_target,_ = reconstruct_tomogram((fringes1[:,:,:]),z=2)
normalized_volume_target[:,:,:], min_vals, range_vals = normalize_volume_by_aline(abs(fftfringes_target))
min_vals_list_target.append(min_vals)
range_vals_list_target.append(range_vals)

# plt.figure()
# plt.imshow(normalized_volume_target[:,:,0], label="complex signal fft")
# plt.legend()
# plt.show()
#%%

t = 8
fig,axs = plt.subplots(1,2)
axs[0].imshow(predicted_volume[:, :, t])
axs[1].imshow(normalized_volume_target[:, :, t])

