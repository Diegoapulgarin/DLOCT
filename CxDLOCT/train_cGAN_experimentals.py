#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint

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

from matplotlib import pyplot
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np

import tensorflow as tf
import tensorflow.image as timg
import tensorflow.keras.backend as K
#%%
def ssim_loss(y_true, y_pred):
    # Calculate SSIM between the reference and predicted images
    y_true = tf.square(tf.abs(y_true[:,:,:,0]+1j*y_true[:,:,:,1]))
    y_pred = tf.square(tf.abs(y_pred[:,:,:,0]+1j*y_pred[:,:,:,1]))
    ssim = tf.reduce_mean(timg.ssim(y_true, y_pred, max_val=1))
    # Calculate the difference from perfect SSIM (1.0)
    diff = 1.0 - ssim
    
    return diff

def power_spectrum_loss(y_true, y_pred):
    # Compute the 2D Fourier transform of the reference and predicted images
    # y_pred = tf.squeeze(y_pred)
    y_true = tf.transpose(y_true, perm=[1, 2, 0,1])
    y_pred = tf.transpose(y_pred, perm=[1, 2, 0,1])
    y_true = tf.cast(y_true, tf.complex64)
    y_pred = tf.cast(y_pred, tf.complex64)
    y_true = y_true[:,:,:,0]+1j*y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,0]+1j*y_pred[:,:,:,1]
    ref_fft = tf.signal.fft2d(tf.cast(y_true, tf.complex64))
    pred_fft = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))
    
    # Compute the power spectra by taking the absolute value squared
    ref_power = tf.square(tf.abs(ref_fft))
    pred_power = tf.square(tf.abs(pred_fft))
    
    # Calculate the difference between the power spectra
    diff = tf.reduce_mean(tf.square(ref_power - pred_power))
    
    return diff

def phase_std_loss(y_true, y_pred):
    # Convert the reference and predicted images to complex64 data type
    y_true = y_true[:,:,:,0]+1j*y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,0]+1j*y_pred[:,:,:,1]
    ref_complex = tf.cast(y_true, tf.complex64)
    pred_complex = tf.cast(y_pred, tf.complex64)
    
    # Compute the phase difference between the complex images
    phase_diff = K.abs(K.angle(ref_complex) - K.angle(pred_complex))
    
    # Calculate the standard deviation of the phase difference
    phase_std = tf.math.reduce_std(phase_diff)
    
    return phase_std

def inverseLogScaleSlices(oldslices, slicesMax, slicesMin):
 
    slices = np.copy(oldslices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[ :, :, 0] + 1j*slices[ :, :, 1])
    slicesPhase = np.angle(slices[ :, :, 0] + 1j*slices[ :, :, 1])
    slicesAmp = slicesAmp * (slicesMax - slicesMin) + slicesMin
    slicesAmp = 10**(slicesAmp)
    slices[ :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[ :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices


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
    e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
    b = Activation('relu')(b)
    # decoder model -> Secuencia de ConvoluciÃ³n y activaciÃ³n -> Decoder descendente
    d1 = decoder_block(b, e6, 512)
    d2 = decoder_block(d1, e5, 512)
    d3 = decoder_block(d2, e4, 512)
    d4 = decoder_block(d3, e3, 256, dropout=False)
    d5 = decoder_block(d4, e2, 128, dropout=False)
    d6 = decoder_block(d5, e1, 64, dropout=False)
    #d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d6)
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
def generate_real_samples(dataset, n_samples, patch_shape,slicesmin,slicesmax):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # slice max and min
    smax = slicesmax[ix]
    smin = slicesmin[ix]
    # generate âœ¬realâœ¬ class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y ,smin,smax
    
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
    [X_realA, X_realB], _,smin,smax = generate_real_samples(dataset, n_samples,
                                                            1,min_vals_tomograms,
                                                            max_vals_tomograms)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # plot real source images
    for i in range(n_samples):
        invslice0 = single_revert_normalization(X_realA[i], smax[i], smin[i])
        plot0 = 10*np.log10(abs(invslice0)**2)
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(plot0, cmap='gray')
    # plot generated target image
    for i in range(n_samples):
        invslice1 = single_revert_normalization(X_fakeB[i], smax[i], smin[i])
        plot1 = 10*np.log10(abs(invslice1)**2)
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(plot1, cmap='gray')
    # plot real target image
    for i in range(n_samples):
        invslice2 = single_revert_normalization(X_realB[i], smax[i], smin[i])
        plot2 = 10*np.log10(abs(invslice2)**2)
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(plot2, cmap='gray')    
        print(smin[i],',',smax[i])
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
        [X_realA, X_realB], y_real,_,_ = generate_real_samples(dataset, n_batch, n_patch,min_vals_tomograms,max_vals_tomograms)
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


def read_tomogram(file_path, dimensions):
    depth, height, width = dimensions
    with open(file_path, 'rb') as file:
        tomogram = np.fromfile(file, dtype='single')
        tomogram = tomogram.reshape((depth, height, width))
    return tomogram

def extract_dimensions(file_name):
    parts = file_name.split('_')
    dimensions = []
    for part in parts:
        if 'z=' in part or 'x=' in part or 'y=' in part:
            number = int(part.split('=')[-1])
            dimensions.append(number)
    return tuple(dimensions)


def normalize_bscans(tomogram):
    # Inicializa listas para almacenar los valores máximos y mínimos de cada B-scan
    min_vals = []
    max_vals = []

    # Inicializa una lista para almacenar los B-scans normalizados
    normalized_bscans = []

    # Itera sobre cada B-scan en el tomograma
    for i in range(tomogram.shape[2]):
        bscan = tomogram[:,:,i]
        # Calcula la magnitud al cuadrado del B-scan
        magnitude_squared = np.abs(bscan) ** 2

        # Aplica logaritmo a la magnitud al cuadrado para obtener la escala logarítmica
        log_magnitude = 10 * np.log10(magnitude_squared + 1e-12)  # Se añade un pequeño valor para evitar logaritmo de cero
        
        # Normaliza los datos entre 0 y 1
        min_val = np.min(log_magnitude)
        max_val = np.max(log_magnitude)
        normalized_bscan = (log_magnitude - min_val) / (max_val - min_val)

        # Almacena los valores máximos y mínimos
        min_vals.append(min_val)
        max_vals.append(max_val)

        # Agrega el B-scan normalizado a la lista
        normalized_bscans.append(normalized_bscan)

    # Convierte las listas en arrays de NumPy
    normalized_bscans = np.array(normalized_bscans)
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)

    return normalized_bscans, min_vals, max_vals




def revert_normalization(normalized_bscans, min_vals, max_vals):

    reverted_bscans = []


    for normalized_bscan, min_val, max_val in zip(normalized_bscans, min_vals, max_vals):

        log_magnitude = normalized_bscan * (max_val - min_val) + min_val
        

        magnitude_squared = 10 ** (log_magnitude / 10)
        

        reverted_bscans.append(magnitude_squared)

    reverted_bscans = np.array(reverted_bscans)

    return reverted_bscans

def single_revert_normalization(bscan,min_val,max_val):
    log_magnitude = bscan[:,:,0] * (max_val - min_val) + min_val
    magnitude_squared = 10 ** (log_magnitude/10)
    xcompare = magnitude_squared*np.exp(1j*bscan[:,:,1])
    return xcompare
    

#%%
base_path = 'C:\\Users\\USER\\Documents\\GitHub\\Experimental_Data_complex'

tissues = ['depth_nail']#, 'depth_fovea', 'depth_opticNerve','depth_chicken']
all_tomograms = []
all_targets = []


for tissue in tissues:
    artifact_path = os.path.join(base_path, 'tomogram_artifacts', tissue)
    no_artifact_path = os.path.join(base_path, 'tomogram_no_artifacts', tissue)

    artifact_files = os.listdir(artifact_path)
    no_artifact_files = os.listdir(no_artifact_path)


    for real_file, imag_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(artifact_path, real_file)
        imag_file_path = os.path.join(artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_tomograms.extend(tom.reshape(-1, dimensions[1], dimensions[2]))


    for real_file, imag_file in zip(no_artifact_files[::2], no_artifact_files[1::2]):
        real_file_path = os.path.join(no_artifact_path, real_file)
        imag_file_path = os.path.join(no_artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_targets.extend(tom.reshape(-1, dimensions[1], dimensions[2]))
    del tom, tomImag, tomReal
    print(tissue, ' loaded')

all_tomograms = np.array(all_tomograms)
all_targets = np.array(all_targets)
# all_tomograms = all_tomograms[:,:,0:10]
# all_targets = all_targets[:,:,0:10]

print(all_tomograms.shape)
print(all_targets.shape)
ampl_tom = abs(all_tomograms)
phase_tom = np.angle(all_tomograms)
ampl_target = abs(all_targets)
phase_target = np.angle(all_targets)
del all_tomograms, all_targets
ampl_tom,min_vals_tomograms,max_vals_tomograms = normalize_bscans(ampl_tom)
ampl_target,min_vals_targets,max_vals_targets = normalize_bscans(ampl_target)
#%%
phase_tom = np.transpose(phase_tom,(2,0,1))
tomCC = np.stack((ampl_tom,phase_tom),axis=3)
#%%
phase_target = np.transpose(phase_target,(2,0,1))
tomTarget = np.stack((ampl_target,phase_target),axis=3)
#%%
dataset = [tomCC,tomTarget]
image_shape = (dimensions[0],dimensions[1],2)

#%%
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)
d_loss1_epoch = []
d_loss2_epoch = []
g_loss_epoch  = []
n_steps_epoch = []
n_epochs = 100
#%%
train(d_model, g_model, gan_model, dataset,n_epochs)

# np.save('/home/dapulgaris/Models/cGAN_1/d_loss1', d_loss1_epoch)
# np.save('/home/dapulgaris/Models/cGAN_1/d_loss2', d_loss2_epoch)
# np.save('/home/dapulgaris/Models/cGAN_1/g_loss',  g_loss_epoch)
# np.save('/home/dapulgaris/Models/cGAN_1/n_epochs', n_epochs)


# bscan = 0
# plot1 = 10*np.log10(abs(all_tomograms[:,:,bscan])**2)
# plot2 = 10*np.log10(abs(all_targets[:,:,bscan])**2)
# fig,ax = plt.subplots(1,2)
# ax[0].imshow(plot1,cmap='gray',vmin=20,vmax=120)
# ax[1].imshow(plot2,cmap='gray',vmin=20,vmax=120)