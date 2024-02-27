#%%
import os
import numpy as np
import tensorflow as tf
from numpy.fft import fft, fft2,fftshift,ifft,ifft2,ifftshift
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

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

def compute_separate_stft_for_volume(volume, fs, nperseg, noverlap):
    """
    Computes the STFT separately for the real and imaginary parts of each A-line in a given volume.
    
    Parameters:
    - volume: A 3D array with dimensions (z, x, bscans), representing the OCT data.
    - fs: Sampling frequency of the A-line data.
    - nperseg: Length of each segment for the STFT.
    - noverlap: Number of points to overlap between segments.
    
    Returns:
    - A 5D array with the STFT results with dimensions (2, freq_bins, time_bins, x, bscans),
      where 2 represents real and imaginary parts, freq_bins is the number of frequency bins,
      and time_bins is the number of time bins.
    """
    # Initialize a list to hold the STFT results for each A-line
    stft_results_real = []
    stft_results_imag = []
    stft_results_complex = []

    # Calculate the STFT for each A-line in the volume
    for bscan_idx in range(volume.shape[2]):
        for a_line_idx in range(volume.shape[1]):
            # Extract the A-line from the volume
            a_line = volume[:, a_line_idx, bscan_idx]

            # Compute the STFT of the real part of the A-line
            _, _, Zxx_real = stft(np.real(a_line), fs=fs, nperseg=nperseg, noverlap=noverlap)
            stft_results_real.append(Zxx_real)

            # Compute the STFT of the imaginary part of the A-line
            _, _, Zxx_imag = stft(np.imag(a_line), fs=fs, nperseg=nperseg, noverlap=noverlap)
            stft_results_imag.append(Zxx_imag)

            # Compute the STFT of the complex A-line
            _, _, Zxx_complex = stft((a_line), fs=fs, nperseg=nperseg, noverlap=noverlap)
            stft_results_complex.append(Zxx_complex)

    # Convert the list of STFT results into a 4D numpy array
    stft_array_real = np.array(stft_results_real)
    stft_array_imag = np.array(stft_results_imag)
    stft_array_complex = np.array(stft_results_complex)

    # Reshape the array to have the correct dimensions
    freq_bins, time_bins = stft_array_real.shape[1], stft_array_real.shape[2]
    stft_array_real = stft_array_real.reshape(volume.shape[2], volume.shape[1], freq_bins, time_bins)
    stft_array_imag = stft_array_imag.reshape(volume.shape[2], volume.shape[1], freq_bins, time_bins)
    freq_bins, time_bins = stft_array_complex.shape[1], stft_array_complex.shape[2]
    stft_array_complex = stft_array_complex.reshape(volume.shape[2], volume.shape[1], freq_bins, time_bins)
    # Combine the real and imaginary STFT results into one array
    stft_combined = np.stack((stft_array_real, stft_array_imag), axis=0)
    
    # Transpose the array to bring it to the desired shape
    stft_combined = stft_combined.transpose(3, 4, 0, 2, 1)
    stft_complex = stft_array_complex.transpose(3, 2, 1, 0,)
    return stft_combined,stft_complex

def basicNormalize(volume):
    n_stft = volume.shape[2]
    mean_vals = np.zeros(n_stft)
    std_vals = np.zeros(n_stft)
    volume_normalized = np.zeros_like(volume)
    for i in range(n_stft):
        mean_vals[i] = np.mean(volume[:, :, i])
        std_vals[i] = np.std(volume[:, :, i])  
        volume_normalized[:, :, i] = (volume[:, :, i] - mean_vals[i]) / std_vals[i]
    return volume_normalized,mean_vals,std_vals


#%%
pathcomplex = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscan'
artifact_files = os.listdir(pathcomplex)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(pathcomplex, real_file)
        imag_file_path = os.path.join(pathcomplex, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        del tomImag, tomReal
fringescc = fftshift(ifft(tom,axis=0),axes=0)
all_tomograms = np.array(fringescc[:,:,4:6])

nperseg = 128
noverlap = 110
fs=279273
_,stft_volume_tomograms = compute_separate_stft_for_volume(all_tomograms[0:2286,:,:],fs, nperseg, noverlap)
dim = np.shape(stft_volume_tomograms)
stft_tomograms = np.reshape(stft_volume_tomograms,(dim[0],dim[1],(dim[2]*dim[3])))
amp_tomograms = abs(stft_tomograms)
phase_tomograms = np.angle(stft_tomograms)
amp_normalized_tomograms,mean_tomograms,std_tomograms = basicNormalize(amp_tomograms)
dim=np.shape(amp_normalized_tomograms)
X = np.zeros((dim[0],dim[1],dim[2],2))
X[:,:,:,0] = np.real(amp_normalized_tomograms*np.exp(1j*phase_tomograms))
X[:,:,:,1] = np.imag(amp_normalized_tomograms*np.exp(1j*phase_tomograms))
X = np.transpose(X,(2,0,1,3))

#%%
modelPath = r'C:\Users\USER\Documents\models\pix2pixstft\pix2pix31'
model = tf.keras.models.load_model(modelPath+'/model_110592.h5')
stft_reconstructed = np.array(model.predict(X, batch_size=8), dtype='float32')
y = stft_reconstructed[:,:,:,0] + 1j*stft_reconstructed[:,:,:,1]
#%%
_, Zoriginal = istft((y[512,:,:]), fs=fs, nperseg=nperseg, noverlap=noverlap)
# plt.imshow(abs(y[512,:,:]))

def reconstruct_volume(spectrogram_volume, fs, nperseg, noverlap, z_dim=2286, x_dim=1024, y_dim=2):
    """
    Reconstructs the original volume from a 3D array of spectrograms.
    
    Parameters:
    - spectrogram_volume: 3D array with dimensions (spectrogramas, x, y), where 'spectrogramas' is the number of STFT results.
    - fs: Sampling frequency.
    - nperseg: Length of each segment for STFT.
    - noverlap: Number of overlapping points between segments.
    - z_dim: The total depth of the reconstructed volume (2286).
    - x_dim: The width of each slice in the reconstructed volume (1024).
    - y_dim: The height of the volume (2).
    
    Returns:
    - A 3D array representing the reconstructed volume with dimensions (z, x, y).
    """
    # Initialize the reconstructed volume
    reconstructed_volume = np.zeros((z_dim, x_dim, y_dim), dtype=np.float)

    # Calculate the number of time steps (spectrogramas) per A-line
    time_steps_per_aline = spectrogram_volume.shape[0] // (x_dim * y_dim)

    for j in range(y_dim):
        for i in range(x_dim):
            # Calculate index in the flat spectrogram volume
            index = j * x_dim + i
            spectrogram = spectrogram_volume[index * time_steps_per_aline:(index + 1) * time_steps_per_aline, :, :]

            # Reshape spectrogram to 2D (freq_bins, time_bins) if necessary
            spectrogram_2d = spectrogram.reshape(spectrogram.shape[1], spectrogram.shape[2])

            # Apply iSTFT
            _, reconstructed_signal = istft(spectrogram_2d, fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            # Fill the corresponding A-line in the reconstructed volume
            reconstructed_volume[:, i, j] = reconstructed_signal[:z_dim]

    return reconstructed_volume
def inverseBasicNormalize(volume_normalized, mean_vals, std_vals):
    """
    Reverts the normalization applied by the basicNormalize function.
    
    Parameters:
    - volume_normalized: The normalized volume, a 3D numpy array.
    - mean_vals: The mean values used for normalization, a 1D numpy array.
    - std_vals: The standard deviation values used for normalization, a 1D numpy array.
    
    Returns:
    - The denormalized (original) volume, a 3D numpy array.
    """
    # Ensure the input volume is a numpy array
    volume_normalized = np.array(volume_normalized)
    n_stft = volume_normalized.shape[2]
    volume_original = np.zeros_like(volume_normalized)
    
    for i in range(n_stft):
        # Denormalize each "slice" of the volume
        volume_original[:, :, i] = (volume_normalized[:, :, i] * std_vals[i]) + mean_vals[i]
    
    return volume_original


#%%
ampy = inverseBasicNormalize(abs(y), mean_tomograms, std_tomograms)
phasey = np.angle(y)
newy= ampy*np.exp(1j*phasey)
reconstructedreal = reconstruct_volume(ampy, fs, nperseg, noverlap, z_dim=2286, x_dim=1024, y_dim=2)
reconstructedimag = reconstruct_volume(phasey, fs, nperseg, noverlap, z_dim=2286, x_dim=1024, y_dim=2)

#%%


fringes = fringescc[0:2286,:,4:6]
fringescompose = np.real(fringes)*np.exp(-0.9j*reconstructedimag)
tomreconstructed =(fft((fringescompose),axis=0))
plt.imshow(20*np.log10(abs(tomreconstructed[:,:,0])),cmap='gray',vmax=90,vmin=70)