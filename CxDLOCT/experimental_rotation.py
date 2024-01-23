#%%
import numpy as np
import os
from numpy.fft import fft, fft2,fftshift,ifft,ifft2,ifftshift
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from sklearn.model_selection import train_test_split
#%%

def fast_reconstruct(array):
    tom = fftshift(fft(fftshift(array,axes=0),axis=0),axes=0)
    return tom

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

    # Convert the list of STFT results into a 4D numpy array
    stft_array_real = np.array(stft_results_real)
    stft_array_imag = np.array(stft_results_imag)

    # Reshape the array to have the correct dimensions
    freq_bins, time_bins = stft_array_real.shape[1], stft_array_real.shape[2]
    stft_array_real = stft_array_real.reshape(volume.shape[2], volume.shape[1], freq_bins, time_bins)
    stft_array_imag = stft_array_imag.reshape(volume.shape[2], volume.shape[1], freq_bins, time_bins)

    # Combine the real and imaginary STFT results into one array
    stft_combined = np.stack((stft_array_real, stft_array_imag), axis=0)
    
    # Transpose the array to bring it to the desired shape
    stft_combined = stft_combined.transpose(3, 4, 0, 2, 1)
    
    return stft_combined

def normalize_magnitude_spectrogram(volume):
    """
    Normalizes the magnitude of each spectrogram in the volume.

    Parameters:
    - volume: A 5D array with dimensions (X, Y, real/imag, alines, bscans).

    Returns:
    - norm_volume: The normalized volume with magnitude of spectrograms.
    - norm_stats: A list of normalization statistics (min and max values) for each spectrogram.
    """
    X, Y, _, alines, bscans = volume.shape
    norm_volume = np.empty((X, Y, 2, alines, bscans), dtype=np.float32)  # 2 channels for magnitude
    norm_stats = []

    # Iterate over each spectrogram in the volume
    for aline_idx in range(alines):
        for bscan_idx in range(bscans):
            for channel in range(2):  # Original volume has 2 channels (real and imag)
                spectrogram = np.abs(volume[:, :, channel, aline_idx, bscan_idx])

                # Normalize the magnitude
                min_val = np.min(spectrogram)
                max_val = np.max(spectrogram)
                norm_spectrogram = (spectrogram - min_val) / (max_val - min_val) if max_val > min_val else spectrogram

                # Store the normalized magnitude
                norm_volume[:, :, channel, aline_idx, bscan_idx] = norm_spectrogram

                # Store normalization statistics
                norm_stats.append((min_val, max_val))

    return norm_volume, norm_stats

def prepare_data_for_training(normalized_volume, target_volume, test_size=0.2):
    """
    Prepares the normalized volume data and corresponding target volume for training.

    Parameters:
    - normalized_volume: A 5D array with dimensions (X, Y, 2, alines, bscans) for input data.
    - target_volume: A 5D array with dimensions (X, Y, 2, alines, bscans) for target data.
    - test_size: Proportion of the dataset to include in the test split.

    Returns:
    - X_train, X_test: Training and test datasets for input data.
    - y_train, y_test: Training and test datasets for target data.
    """
    # Number of samples
    num_samples = normalized_volume.shape[3] * normalized_volume.shape[4]

    # Reshape the volumes into sets of 3D arrays
    X = normalized_volume.transpose(3, 4, 0, 1, 2).reshape(num_samples, *normalized_volume.shape[:3])
    y = target_volume.transpose(3, 4, 0, 1, 2).reshape(num_samples, *target_volume.shape[:3])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

#%%

base_path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Complex conjugate artifacts data\Experimental_Data_complex'

tissues = ['depth_nail']#,'depth_chicken_breast','depth_nail_2','depth_chicken_breast2']
all_tomograms = []
all_targets = []
bscans = 5
for tissue in tissues:
    print(tissue, ' loading')
    artifact_path = os.path.join(base_path, 'tomogram_artifacts', tissue)
    no_artifact_path = os.path.join(base_path, 'tomogram_no_artifacts', tissue)

    artifact_files = os.listdir(artifact_path)
    no_artifact_files = os.listdir(no_artifact_path)


    for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(artifact_path, real_file)
        imag_file_path = os.path.join(artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_tomograms.append(tom[:,:,0:bscans])
        del tom, tomImag, tomReal


    for imag_file, real_file in zip(no_artifact_files[::2], no_artifact_files[1::2]):
        real_file_path = os.path.join(no_artifact_path, real_file)
        imag_file_path = os.path.join(no_artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_targets.append(tom[:,:,0:bscans])
    del tom, tomImag, tomReal
    print(tissue, ' loaded')

all_tomograms = np.array(all_tomograms)
all_targets = np.array(all_targets)
fringes_tomograms = fftshift(fft(fftshift(all_tomograms,axes=0),axis=0),axes=0)
fringes_targets = fftshift(fft(fftshift(all_targets,axes=0),axis=0),axes=0)
del all_tomograms, all_targets
#%%
# nperseg = 64  # Por ejemplo, si has decidido que este es un buen tamaño de segmento
# noverlap = int(nperseg * 0.75)  # 75% de solapamiento
# fs=279273
# f, t, Zxx1 = stft(np.real(fringes_tomograms[0,:,0,0]), fs=fs, nperseg=nperseg, noverlap=noverlap)
# f, t, Zxx2 = stft(np.real(fringes_targets[0,:,0,0]), fs=fs, nperseg=nperseg, noverlap=noverlap)
# fig,axs = plt.subplots(1,2)
# axs[0].imshow(abs(Zxx1)**2)
# # axs[0].set_axis('off')
# axs[0].set_title('artifacts')
# axs[1].imshow(abs(Zxx2)**2)
# # axs[1].set_axis('off')
# axs[1].set_title('no artifacts')
# #%%
# plt.imshow(10*np.log10(abs(all_targets[0,:,:,0])**2))
# t, A_line_reconstructed = istft(Zxx2, fs=fs, nperseg=nperseg, noverlap=noverlap)
# plt.plot(A_line_reconstructed)
#%%

nperseg = 64
noverlap = int(nperseg * 0.75)
fs=279273
hacer listas
for i in range(np.shape(fringes_targets[0])):
    stft_volume = compute_separate_stft_for_volume(fringes_targets[i,:,:,:],fs, nperseg, noverlap)
    normalized_volume, normalization_stats = normalize_magnitude_spectrogram(stft_volume)


#%%


# Example usage (assuming normalized_volume and target_volume are defined and have the correct shape):
X_train, X_test, y_train, y_test = prepare_data_for_training(normalized_volume, target_volume)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Shapes of the training and test datasets

#%%
t, A_line_reconstructed = istft(stft_volume[0,0,:,0,:], fs=fs, nperseg=nperseg, noverlap=noverlap)
plt.plot(A_line_reconstructed)
#%%

