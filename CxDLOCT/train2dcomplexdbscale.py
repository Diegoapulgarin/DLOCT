#%%
import numpy as np 
import os
import sys
# sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\Analysis_cGAN')
# from Deep_Utils import dbscale
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift,ifft
from tqdm import tqdm

from os import sep
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

def dbscale(darray):
    if len(np.shape(darray))==3:
        img = 10*np.log10(abs(darray[:,:,0]+1j*darray[:,:,1])**2)
    else:
        img = 10*np.log10(abs(darray[:,:])**2)
    return img


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

def read_tomogram2(file_path, dimensions):
    depth, height, width = dimensions
    with open(file_path, 'rb') as file:
        tomogram = np.fromfile(file, dtype='single')
        tomogram = tomogram.reshape((depth, height, width))
    return tomogram

def logScale(slices):
    
    logslices = np.copy(slices)
    nSlices = len(slices)
    logslicesAmp = abs(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # and retrieve the phase
    logslicesPhase = np.angle(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # reescale amplitude
    logslicesAmp = np.log10(logslicesAmp)
    slicesMax = np.reshape(logslicesAmp.max(axis=(1, 2)), (nSlices, 1, 1))
    slicesMin = np.reshape(logslicesAmp.min(axis=(1, 2)), (nSlices, 1, 1))
    logslicesAmp = (logslicesAmp - slicesMin) / (slicesMax - slicesMin)
    # --- here, we could even normalize each slice to 0-1, keeping the original
    # --- limits to rescale after the network processes
    # and redefine the real and imaginary components with the new amplitude and
    # same phase
    logslices[:, :, :, 0] = (np.real(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    logslices[:, :, :, 1] = (np.imag(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    
    return logslices, slicesMax, slicesMin

def inverseLogScale(oldslices, slicesMax, slicesMin):
 
    slices = np.copy(oldslices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesPhase = np.angle(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesAmp = slicesAmp * (slicesMax - slicesMin) + slicesMin
    slicesAmp = 10**(slicesAmp)
    slices[:, :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices

def normalize_tomogram(tomogram):
    z, x, y, _ = tomogram.shape
    normalized_tomogram = np.zeros_like(tomogram)
    max_values = np.zeros((y, 2))
    min_values = np.zeros((y, 2))
    c = 0.0001
    tomogram = 10*np.log10(abs(tomogram+c)**2)
    for i in range(y):
        for j in range(2):  # Real and imaginary parts
            bscan = tomogram[:, :, i, j]
            max_val = np.max(bscan)
            min_val = np.min(bscan)
            normalized_tomogram[:, :, i, j] = (bscan - min_val) / (max_val - min_val)
            max_values[i, j] = max_val
            min_values[i, j] = min_val
    return normalized_tomogram, max_values, min_values

def denormalize_tomogram(normalized_tomogram, max_values, min_values):
    z, x, y, _ = normalized_tomogram.shape
    denormalized_tomogram = np.zeros_like(normalized_tomogram)
    c = 0.0001
    for i in range(y):
        for j in range(2):  # Real and imaginary parts
            bscan_normalized = normalized_tomogram[:, :, i, j]
            max_val = max_values[i, j]
            min_val = min_values[i, j]
            denormalized_tomogram[:, :, i, j] = bscan_normalized * (max_val - min_val) + min_val
    denormalized_tomogram = np.sqrt(10**(denormalized_tomogram/10))-c
    return denormalized_tomogram

def denormalize_bscan(normalized_tomogram, max_values, min_values):
    denormalized_tomogram = np.zeros_like(normalized_tomogram)
    c = 0.0001
    for j in range(2):  # Real and imaginary parts
        bscan_normalized = normalized_tomogram[:, :, j]
        max_val = max_values[j]
        min_val = min_values[j]
        denormalized_tomogram[:, :, j] = bscan_normalized * (max_val - min_val) + min_val
        denormalized_tomogram = np.sqrt(10**(denormalized_tomogram/10))-c
    return denormalized_tomogram


#%% custom loss function

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

# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=42)
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
    init = RandomNormal(stddev=0.02, seed=42)
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
    init = RandomNormal(stddev=0.02, seed=42)
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
    init = RandomNormal(stddev=0.02, seed=42)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model -> Secuencia de ConvoluciÃ³n y activaciÃ³n -> Encoder ascendente
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 256)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model -> Secuencia de ConvoluciÃ³n y activaciÃ³n -> Decoder descendente
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 256, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
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
    smax = slicesmax[ix,:,:]
    smin = slicesmin[ix,:,:]
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
    path = 'C:/Users/USER/Documents/models'
    # select a sample of input images
    [X_realA, X_realB], _,smin,smax = generate_real_samples(dataset, n_samples,
                                                            1,combined_max,
                                                            combined_min)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    vmin=60
    vmax=120
    # plot real source images
    for i in range(n_samples):
        invslice0 = denormalize_bscan(X_realA[i], smax[i], smin[i])
        plot0 = 10*np.log10(abs(invslice0[:,:,0]+1j*invslice0[:,:,1])**2)
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(plot0, cmap='hot', vmin=vmin, vmax=vmax)
    # plot generated target image
    for i in range(n_samples):
        invslice1 = denormalize_bscan(X_fakeB[i], smax[i], smin[i])
        plot1 = 10*np.log10(abs(invslice1[:,:,0]+1j*invslice1[:,:,1])**2)
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(plot1, cmap='hot', vmin=vmin, vmax=vmax)
    # plot real target image
    for i in range(n_samples):
        invslice2 = denormalize_bscan(X_realB[i], smax[i], smin[i])
        plot2 = 10*np.log10(abs(invslice2[:,:,0]+1j*invslice2[:,:,1])**2)
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(plot2, cmap='hot', vmin=vmin, vmax=vmax)    
        print(smin[i],',',smax[i])
    # save plot to file
    filename1 = path +'/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = path+'/model_%06d.h5' % (step+1)
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
        [X_realA, X_realB], y_real,_,_ = generate_real_samples(dataset, n_batch, n_patch,combined_max,combined_min)
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
#%% reading experimental tomogram
dataPath = r'E:\DLOCT\Experimental_Data_complex'
noArtifacts = 'tomogram_no_artifacts'
artifacts = 'tomogram_artifacts'
folder = ['depth_nail']

pathcomplex = os.path.join(dataPath,artifacts,folder[0])
artifact_files = os.listdir(pathcomplex)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(pathcomplex, real_file)
        imag_file_path = os.path.join(pathcomplex, imag_file)
        print(f'real: {real_file_path}')
        print(f'imag: {imag_file_path}')
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram2(real_file_path, dimensions)
        tomImag = read_tomogram2(imag_file_path, dimensions)
        tomcc = np.stack((tomReal,tomImag),axis=3)
        del tomImag, tomReal


pathcomplex = os.path.join(dataPath,noArtifacts,folder[0])
artifact_files = os.listdir(pathcomplex)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(pathcomplex, real_file)
        imag_file_path = os.path.join(pathcomplex, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram2(real_file_path, dimensions)
        tomImag = read_tomogram2(imag_file_path, dimensions)
        tom = np.stack((tomReal,tomImag),axis=3)
        del tomImag, tomReal
#% #################################### for preprocessing testing, delete after #####################################
tom = tom[:,:,250:258,:]
tomcc = tomcc[:,:,250:258,:]
#%%reading sinthetic tomograms
dataPath = r'C:\Users\USER\Documents\GitHub\Simulated_Data_Complex\tom1'
sinteticlist = os.listdir(dataPath)
tomTarget = []
tomInput = []
tomTarget.append(tom)
tomInput.append(tomcc)
for tomName in tqdm(sinteticlist):
     if tomName.split(sep='_')[1] == 'imag':
        name = tomName.split(sep='_')
        dimensions = extract_dimensions(tomName[:-4])
        loadName = '_'.join([name[0],'real',name[2],name[3],name[4],name[5]])
        real_file_path = os.path.join(dataPath,loadName)
        imag_file_path = os.path.join(dataPath,tomName)
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tomsyntetic = np.stack((tomReal,tomImag),axis=3)
        tomsyntetic = np.pad(tomsyntetic,pad_width=((256,256),(0,0),(0,0),(0,0)),mode='constant',constant_values=1)
        tomsyntetic = np.pad(tomsyntetic,pad_width=((0,0),(256,256),(0,0),(0,0)),mode='reflect')
        tomccsyntetic = ifft(tomsyntetic[:,:,:,0]+1j*tomsyntetic[:,:,:,1],axis=0)
        tomccsyntetic = fft(tomccsyntetic.real,axis=0)
        tomccsyntetic = np.stack((tomccsyntetic.real,tomccsyntetic.imag),axis=3)
        tomTarget.append(tomsyntetic)
        tomInput.append(tomccsyntetic)
#%% normalization amplitude and phase process process
tomTargetNorm = []
tomInputNorm = []
tomTargetmax =[]
tomTargetmin =[]
tomInputmax =[]
tomInputmin =[]
for t in tqdm(range(len(tomInput))):
    tomNorm,tmax,tmin = normalize_tomogram(tomTarget[t])
    tomccNorm,imax,imin = normalize_tomogram(tomInput[t])
    tomTargetNorm.append(tomNorm)
    tomInputNorm.append(tomccNorm)
    tomTargetmax.append(tmax)
    tomTargetmin.append(tmin)
    tomInputmax.append(imax)
    tomInputmin.append(imin)

#%%
def combine_tomograms(tomogram_list):
    z, x, y, _ = tomogram_list[0].shape
    combined_tomograms = []
    for tomogram in tomogram_list:
        # Re-order (y, z, x, 2)
        rearranged_tomogram = np.transpose(tomogram, (2, 0, 1, 3))
        combined_tomograms.append(rearranged_tomogram)
    combined_array = np.concatenate(combined_tomograms, axis=0)
    return combined_array

combined_target = combine_tomograms(tomTargetNorm)
combined_input = combine_tomograms(tomInputNorm)
combined_input_max = np.concatenate(tomInputmax, axis=0)
combined_input_min = np.concatenate(tomInputmin, axis=0)
combined_target_max = np.concatenate(tomTargetmax, axis=0)
combined_target_min = np.concatenate(tomTargetmin, axis=0)
combined_max = np.stack((combined_target_max,combined_input_max),axis=2)
combined_min = np.stack((combined_target_min,combined_input_min),axis=2)
print(combined_target.shape)

image_shape = (combined_target.shape[1],combined_target.shape[2],2)
#%%
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
dataset=[combined_input[0:3,:,:,:],combined_target[0:3,:,:,:]]

# %%
# def denormalize_bscan(normalized_tomogram, max_values, min_values):
#     denormalized_tomogram = np.zeros_like(normalized_tomogram)
#     for j in range(2):  # Real and imaginary parts
#         bscan_normalized = normalized_tomogram[:, :, j]
#         max_val = max_values[j]
#         min_val = min_values[j]
#         denormalized_tomogram[:, :, j] = bscan_normalized * (max_val - min_val) + min_val
#     return denormalized_tomogram

# n_samples = 3
# step = 1
# path = r'C:\Users\USER\Documents\GitHub\imagenes test'
# # select a sample of input images
# [X_realA, X_realB], _,smin,smax = generate_real_samples(dataset, n_samples,
#                                                         1, combined_min,combined_max)
# # generate a batch of fake samples
# X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

# vmin=60
# vmax=120
# # plot real source images
# for i in range(n_samples):
#     # invslice0 = denormalize_bscan(X_realA[i], smax[i,:,1], smin[i,:,1])
#     plot0 = (abs((X_realA[i,:,:,0]+1j*X_realA[i,:,:,1])))
#     pyplot.subplot(3, n_samples, 1 + i)
#     pyplot.axis('off')
#     pyplot.imshow(plot0, cmap='hot')

# # plot generated target image
# for i in range(n_samples):
#     # invslice1 = denormalize_bscan(X_fakeB[i], smax[i,:,1], smin[i,:,1])
#     plot1 = (abs(X_fakeB[i,:,:,0]+1j*X_fakeB[i,:,:,1]))
#     pyplot.subplot(3, n_samples, 1 + n_samples + i)
#     pyplot.axis('off')
#     pyplot.imshow(plot1, cmap='hot')
# # plot real target image
# for i in range(n_samples):
#     # invslice2 = denormalize_bscan(X_realB[i], smax[i,:,0], smin[i,:,0])
#     plot2 = (abs((X_realB[i,:,:,0]+1j*X_realB[i,:,:,1])))
#     pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
#     pyplot.axis('off')
#     pyplot.imshow(plot2, cmap='hot')
# # save plot to file
# filename1 = path +'/plot_%06d.png' % (step+1)
# pyplot.savefig(filename1)
# pyplot.close()
# # save the generator model
# opt = RMSprop(learning_rate=0.0002)
# g_model.compile(loss=['mean_squared_error', 'mae'], optimizer=opt, loss_weights=None)
# filename2 = path+'/model_%06d.h5' % (step+1)
# g_model.save(filename2)


#%%
# train model
d_loss1_epoch = []
d_loss2_epoch = []
g_loss_epoch  = []
n_steps_epoch = []
n_epochs = 100
train(d_model, g_model, gan_model, dataset,n_epochs)

# np.save('/home/dapulgaris/Models/cGAN_1/d_loss1', d_loss1_epoch)
# np.save('/home/dapulgaris/Models/cGAN_1/d_loss2', d_loss2_epoch)
# np.save('/home/dapulgaris/Models/cGAN_1/g_loss',  g_loss_epoch)
# np.save('/home/dapulgaris/Models/cGAN_1/n_epochs', n_epochs)