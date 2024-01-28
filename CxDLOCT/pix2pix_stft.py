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
nperseg = 127
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
shape = np.shape(X_train)
input_shape = (shape[1],shape[2],shape[3])
print(input_shape)
#%%
from tensorflow.keras import layers, Model
import tensorflow as tf

def plot_generated_images(epoch, generator, X_data, y_data, examples=3, figsize=(12, 4),pathToSave=''):
    # Seleccionar muestras aleatorias
    
    idx = np.random.randint(0, X_data.shape[0], examples)
    x_samples = X_data[idx]
    y_samples = y_data[idx]
    
    generated_images = generator.predict(x_samples)

    for i in range(examples):
        # Combinar las partes real e imaginaria para formar el espectrograma
        generated_spectrogram = np.abs(generated_images[i, :, :, 0] + 1j * generated_images[i, :, :, 1])
        real_spectrogram = np.abs(y_samples[i, :, :, 0] + 1j * y_samples[i, :, :, 1])

        plt.figure(figsize=figsize)

        # Visualización de la entrada
        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(x_samples[i, :, :, 0] + 1j * x_samples[i, :, :, 1]), cmap='gray')
        plt.title('Input')

        # Visualización de la imagen generada
        plt.subplot(1, 3, 2)
        plt.imshow(generated_spectrogram, cmap='gray')
        plt.title('Generated')

        # Visualización de la imagen objetivo
        plt.subplot(1, 3, 3)
        plt.imshow(real_spectrogram, cmap='gray')
        plt.title('Real')

        plt.tight_layout()
        plt.savefig(os.path.join(pathToSave,f'gan_generated_image_epoch_{epoch+1}_example_{i}.png'))
        plt.close()


def build_generator(input_shape=(64, 80, 2)):
    inputs = layers.Input(shape=input_shape)

    # Encoder: Capas de downsampling
    down1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    down1_pool = layers.MaxPooling2D((2, 2), strides=2)(down1)

    # Aquí podrías agregar más capas de downsampling si es necesario

    # Decoder: Capas de upsampling
    up1 = layers.UpSampling2D((2, 2))(down1_pool)
    up1_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)

    # Aquí podrías agregar más capas de upsampling si es necesario

    # Salida del generador
    outputs = layers.Conv2D(2, (1, 1), activation='tanh')(up1_conv)

    return Model(inputs=inputs, outputs=outputs)

def build_discriminator(input_shape=(64, 80, 4)):
    inputs = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    conv1_leaky = layers.LeakyReLU(alpha=0.2)(conv1)

    # Aquí podrías agregar más capas convolucionales si es necesario

    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv1_leaky)

    return Model(inputs=inputs, outputs=outputs)

# input_shape = (256, 256, 2) # Ajustar según las dimensiones de tus espectrogramas

# Construir el generador y el discriminador
generator = build_generator()
discriminator = build_discriminator()

# Compilar el discriminador
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Compilar la GAN completa
gan_input = tf.keras.layers.Input(shape=(64, 80, 2))
fake_image = generator(gan_input)
discriminator.trainable = False
# Concatena la imagen de entrada con la imagen generada
combined_images = tf.keras.layers.Concatenate(axis=-1)([gan_input, fake_image])

# Configura el discriminador para que no sea entrenable
discriminator.trainable = False

# La salida de la GAN es la salida del discriminador
gan_output = discriminator(combined_images)

gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Parámetros de entrenamiento
epochs = 30
batch_size = 32
visualization = 10
pathToSave = r'C:\Users\USER\Documents\models\pix2pixstft'
# Bucle de entrenamiento
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    for i in range(0, len(X_train), batch_size):
        # Obtener lote de datos
        X_real = X_train[i:i + batch_size]
        y_real = y_train[i:i + batch_size]
        
        # Generar imágenes falsas
        y_fake = generator.predict(X_real)

        # Concatenar la entrada y la salida real para el discriminador
        input_real = np.concatenate([X_real, y_real], axis=-1)

        # Concatenar la entrada y la salida generada para el discriminador
        input_fake = np.concatenate([X_real, y_fake], axis=-1)

        # Etiquetas para datos reales y falsos
        discriminator_output_shape = discriminator.output_shape[1:]

# Crear etiquetas para datos reales y falsos con el tamaño de salida del discriminador
        real_labels = np.ones((batch_size, *discriminator_output_shape))
        fake_labels = np.zeros((batch_size, *discriminator_output_shape))

        # Entrenar el discriminador
        discriminator_loss_real = discriminator.train_on_batch(input_real, real_labels)
        discriminator_loss_fake = discriminator.train_on_batch(input_fake, fake_labels)
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Etiquetas para engañar al discriminador
        # Crear trick_labels para engañar al discriminador
        # Ajustar las dimensiones de trick_labels para coincidir con la salida del discriminador
        trick_labels = np.ones((batch_size, 32, 40, 1))  # Asegúrate de que tenga 4 dimensiones

  # Ajustar las dimensiones para coincidir con la salida del discriminador


        # Entrenar el generador
        generator_loss = gan.train_on_batch(X_real,trick_labels)

        # Progreso del entrenamiento
        print(f"Batch {i//batch_size}: [Discriminator loss: {discriminator_loss}] [Generator loss: {generator_loss}]")
    if (epoch + 1) % visualization == 0:
        plot_generated_images(epoch, generator, X_train, y_train)
        generator.save(os.path.join(pathToSave,f'generator_epoch_{epoch+1}.h5'))

    # Opcional: Guardar el modelo al final de cada época
    # generator.save('generator_epoch_{epoch}.h5')
#%%