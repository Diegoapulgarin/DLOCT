#%%
import numpy as np
import scipy.io as sio
import os
from scipy.fft import fft, fftshift
from numpy.random import randn
#%%
path = r'C:\Users\USER\Documents\GitHub\Simulated_Data_Complex'
os.chdir(path)
fringes = []
for filename in os.listdir(os.getcwd()):
   print(path+'/'+filename)
   mat_contents = sio.loadmat(path+'/'+filename)
   fringes1 = mat_contents['fringes1']
   divisions = int(fringes1.shape[2]/16)
   n = 0 
   for i in range(divisions):
       fringes_slice = fringes1[:, :, n:n+16]
       n = n + 16
       fringes.append(fringes_slice)
   print(filename)
fringes = np.array(fringes)
del fringes1, fringes_slice
#%%


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


def paired_random_zero_padding(tomogram, target, target_z_size=1024):
    assert tomogram.shape == target.shape, "Ambos volúmenes deben tener la misma forma."

    z, x, y = tomogram.shape
    
    # Si el volumen ya es del tamaño deseado o mayor, lo truncamos.
    if z >= target_z_size:
        return tomogram[:target_z_size, :, :], target[:target_z_size, :, :]

    padding_size = target_z_size - z

    # Decidimos aleatoriamente si el padding va al principio, al final, o se divide.
    decision = np.random.choice(["start", "end", "both"])
    
    if decision == "start":
        start_padding = padding_size
        end_padding = 0
    elif decision == "end":
        start_padding = 0
        end_padding = padding_size
    else: # decision == "both"
        start_padding = np.random.randint(0, padding_size + 1)
        end_padding = padding_size - start_padding
    
    # Aplicamos el padding a ambos volúmenes.
    padded_tomogram = np.pad(tomogram, ((start_padding, end_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_target = np.pad(target, ((start_padding, end_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

    return padded_tomogram, padded_target


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
normalized_volume_complex = np.zeros(np.shape(fringes))
min_vals_list =[]
range_vals_list =[]
for i in range(np.shape(fringes)[0]):
    fftfringes,_ = reconstruct_tomogram(fringes[i,:,:,:],z=2)
    normalized_volume_complex[i,:,:,:], min_vals, range_vals = normalize_volume_by_aline(abs(fftfringes))
    min_vals_list.append(min_vals)
    range_vals_list.append(range_vals)

normalized_volume_real = np.zeros(np.shape(fringes))
min_vals_list =[]
range_vals_list =[]
for i in range(np.shape(fringes)[0]):
    fftfringes,_ = reconstruct_tomogram(np.real(fringes[i,:,:,:]),z=2)
    normalized_volume_real[i,:,:,:], min_vals, range_vals = normalize_volume_by_aline(abs(fftfringes))
    min_vals_list.append(min_vals)
    range_vals_list.append(range_vals)

#%%
ntom=np.shape(normalized_volume_complex)[0]
zsize=1024
xsize=np.shape(normalized_volume_complex)[2]
ysize=np.shape(normalized_volume_complex)[3]
padded_tomogram = np.zeros((ntom,zsize,xsize,ysize))
padded_target = np.zeros((ntom,zsize,xsize,ysize))
for i in range(np.shape(padded_target)[0]):
    padded_tomogram[i,:,:,:], padded_target[i,:,:,:] = paired_random_zero_padding(normalized_volume_complex[2,:,:,:], 
                                                                                  normalized_volume_real[2,:,:,:], target_z_size=zsize)


#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(padded_target[:,1,1], label="Single Aline FFT")
plt.plot(padded_tomogram[:, 1, 1], label="Volume FFT")
plt.legend()
plt.show()
