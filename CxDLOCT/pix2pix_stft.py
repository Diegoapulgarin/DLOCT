import numpy as np
import os
from numpy.fft import fft, fft2,fftshift,ifft,ifft2,ifftshift
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import ReLU
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from matplotlib import pyplot
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
zlen = 1000
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
        all_tomograms.append(tom[0:zlen,:,0:bscans])
        del tom, tomImag, tomReal
    for imag_file, real_file in zip(no_artifact_files[::2], no_artifact_files[1::2]):
        real_file_path = os.path.join(no_artifact_path, real_file)
        imag_file_path = os.path.join(no_artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_targets.append(tom[0:zlen,:,0:bscans])
    del tom, tomImag, tomReal
    print(tissue, ' loaded')
all_tomograms = np.array(all_tomograms)
all_targets = np.array(all_targets)
fringes_tomograms = fftshift(fft(fftshift(all_tomograms,axes=0),axis=0),axes=0)
fringes_targets = fftshift(fft(fftshift(all_targets,axes=0),axis=0),axes=0)
del all_tomograms, all_targets
#%%
nperseg = 127
noverlap = int(nperseg * 0.88)
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
image_shape = (shape[1],shape[2],shape[3])
print(image_shape)
#%%
# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = ReLU()(d)#LeakyReLU(alpha=0.2)(d) #LeakyReLU
    
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('relu')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(learning_rate=0.0000001)#Adam , beta_1=0.5 , RMSprop
    model.compile(loss=['binary_crossentropy'], optimizer=opt, loss_weights=None)#'binary_crossentropy'
    
    return model
    
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
    
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model -> Secuencia de ConvoluciÃ³n y activaciÃ³n -> Encoder ascendente
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    # e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    # decoder model -> Secuencia de ConvoluciÃ³n y activaciÃ³n -> Decoder descendente
    d1 = decoder_block(b, e5, 512)
    d2 = decoder_block(d1, e4, 512)
    d3 = decoder_block(d2, e3, 512)
    d4 = decoder_block(d3, e2, 256, dropout=False)
    d5 = decoder_block(d4, e1, 128, dropout=False)
    # d6 = decoder_block(d5, e1, 64, dropout=False)
    #d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
    out_image = Activation('relu')(g) #'tanh'
    # define model
    model = Model(in_image, out_image)
    return model
    
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = RMSprop(learning_rate=0.0002)
    model.compile(loss=['mean_squared_error', 'mae'], optimizer=opt, loss_weights=None) # =['mean_squared_error', 'mae']
    return model
    
# load and prepare training images
def load_real_samples(filename):
    # load the compressed arrays
    data = load(filename)
    # unpack the arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]
    
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # slice max and min
    # generate âœ¬realâœ¬ class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y 
    
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create âœ¬fakeâœ¬ class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples,1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # plot real source images
    for i in range(n_samples):
        spectogram = X_realA[i]
        plot0 = spectogram[:,:,0]+1j*spectogram[:,:,1]
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(abs(plot0), cmap='gray')
    # plot generated target image
    for i in range(n_samples):
        spectogram = X_fakeB[i]
        plot1 = spectogram[:,:,0]+1j*spectogram[:,:,1]
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(abs(plot1), cmap='gray')
    # plot real target image
    for i in range(n_samples):
        spectogram = X_realB[i]
        plot2 = spectogram[:,:,0]+1j*spectogram[:,:,1]
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(abs(plot2), cmap='gray')
    # save plot to file
    filename1 = 'C:/Users/USER/Documents/GitHub/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'C:/Users/USER/Documents/GitHub/model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    d_loss1_val = []
    d_loss2_val = []
    g_loss_val  = []
    n_steps_val = []
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d/%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1,n_steps, d_loss1, d_loss2, g_loss))
        d_loss1_val.append(d_loss1)
        d_loss2_val.append(d_loss2)
        g_loss_val.append(g_loss)
        
        #summarize model performance
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, dataset)
            # calculate mean of losses in epoch
            d_loss1_mean = np.array(d_loss1_val).mean()
            d_loss2_mean = np.array(d_loss2_val).mean()
            g_loss_mean = np.array(g_loss_val).mean()
            # concatenate losses values
            d_loss1_epoch.append(d_loss1_mean)
            d_loss2_epoch.append(d_loss2_mean)
            g_loss_epoch.append(g_loss_mean)
            n_steps_epoch.append(int(i/bat_per_epo))
            # reset losses values
            d_loss1_val = []
            d_loss2_val = []
            g_loss_val  = []

#%%
dataset = [X_train,y_train]
#%%
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)
d_loss1_epoch = []
d_loss2_epoch = []
g_loss_epoch  = []
n_steps_epoch = []
n_epochs = 10
#%%
train(d_model, g_model, gan_model, dataset,n_epochs)