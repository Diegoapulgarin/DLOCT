#%%
import numpy as np
import os
from numpy.fft import fft, fft2,fftshift,ifft,ifft2,ifftshift
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.backend import clear_session
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError, LogCosh
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
nperseg = 128
noverlap = int(nperseg * 0.9)
fs=279273
normalized_stft_tomograms = []
normalized_stft_targets = []
stats_tomograms = []
stats_targets = []
for i in range(np.shape(fringes_targets)[0]):
    stft_volume_tomograms = compute_separate_stft_for_volume(fringes_tomograms[i,:,:,:],fs, nperseg, noverlap)
    stft_volume_targets = compute_separate_stft_for_volume(fringes_targets[i,:,:,:],fs, nperseg, noverlap)
    normalized_volume_tomograms, normalization_stats_tomograms = normalize_magnitude_spectrogram(stft_volume_tomograms)
    normalized_volume_targets, normalization_stats_targets = normalize_magnitude_spectrogram(stft_volume_targets)
    normalized_stft_tomograms.append(normalized_volume_tomograms)
    normalized_stft_targets.append(normalized_volume_targets)
    stats_tomograms.append(normalization_stats_tomograms)
    stats_targets.append(normalization_stats_targets)
normalized_stft_tomograms = np.array(normalized_stft_tomograms)
normalized_stft_targets = np.array(normalized_stft_targets)
stats_tomograms = np.array(stats_tomograms)
stats_targets = np.array(stats_targets)
del stft_volume_tomograms, stft_volume_targets
del normalized_volume_tomograms, normalized_volume_targets
del normalization_stats_targets,normalization_stats_tomograms
# del fringes_targets,fringes_tomograms
X = np.transpose(normalized_stft_tomograms,(1,2,3,4,5,0))
dim = np.shape(X)
X = np.reshape(X,(dim[0],dim[1],dim[2],dim[3],(dim[4]*dim[5])))
Y = np.transpose(normalized_stft_targets,(1,2,3,4,5,0))
dim = np.shape(Y)
Y = np.reshape(Y,(dim[0],dim[1],dim[2],dim[3],(dim[4]*dim[5])))
del normalized_stft_targets,normalized_stft_tomograms
X_train, X_test, y_train, y_test = prepare_data_for_training(X, Y)
#%%
shape = np.shape(X_train)


clear_session()

def fcn_spectrogram_model(input_shape):
    inputs = Input(input_shape)

    # Capa 1
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # Capa 2
    conv2 = Conv2D(128, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(256, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(512, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    # Más capas convolucionales según sea necesario

    # Capa de salida para reconstruir el espectrograma
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(conv2)

    model = Model(inputs=inputs, outputs=outputs)
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

# Crear el modelo
fcn_model = fcn_spectrogram_model((shape[1],shape[2],shape[3]))


# Definir un programador de tasa de aprendizaje que disminuya con el tiempo
def lr_schedule(epoch, lr):
    if epoch > 1:
        lr = lr / 2
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# # MSE + Adam 
# fcn_model.compile(
#     optimizer=Adam(learning_rate=0.001),  # Puedes ajustar la tasa de aprendizaje inicial aquí
#     loss='mean_squared_error',  # O cambiar a 'mean_absolute_error' o 'logcosh'
#     metrics=['accuracy'])
# MSE + SGD
fcn_model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss=MeanSquaredError(),
    metrics=['accuracy'])

# # Huber + Adam
# fcn_model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss=Huber(delta=1.0),
#     metrics=['accuracy'])

# # MAE + RMSprop
# fcn_model.compile(
#     optimizer=RMSprop(learning_rate=0.001),
#     loss=MeanAbsoluteError(),
#     metrics=['accuracy'])

# # Log Cosh + Nadam
# fcn_model.compile(
#     optimizer=Nadam(learning_rate=0.001),
#     loss=LogCosh(),
#     metrics=['accuracy'])

fcn_model.summary()
# Entrenamiento del modelo con el programador de tasa de aprendizaje
history = fcn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16,
    callbacks=[lr_scheduler]  # Agregar el programador de LR a los callbacks
)

#%%

def plot_inference(model, X_test, y_test, num_samples=5):
    predictions = model.predict(X_test[:num_samples])
    
    for i in range(num_samples):
        plt.figure(figsize=(12, 4))

        # Recomponer el espectrograma de la entrada
        input_spectrogram = X_test[i, :, :, 0] + 1j * X_test[i, :, :, 1]
        
        # Recomponer el espectrograma de la salida real
        real_spectrogram = y_test[i, :, :, 0] + 1j * y_test[i, :, :, 1]

        # Recomponer el espectrograma de la predicción
        predicted_spectrogram = predictions[i, :, :, 0] + 1j * predictions[i, :, :, 1]

        # Muestra la entrada
        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(input_spectrogram), cmap='gray')
        plt.title("Entrada")

        # Muestra la salida real
        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(real_spectrogram), cmap='gray')
        plt.title("Salida Real")

        # Muestra la predicción del modelo
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(predicted_spectrogram), cmap='gray')
        plt.title("Predicción del Modelo")

        plt.show()

# Ejemplo de uso
plot_inference(fcn_model, X_test, y_test)



# t, A_line_reconstructed = istft(stft_volume[0,0,:,0,:], fs=fs, nperseg=nperseg, noverlap=noverlap)
# plt.plot(A_line_reconstructed)
#%%

