#%%
import os
import numpy as np
from numpy.random import randint
from matplotlib import pyplot
from keras.initializers import RandomNormal
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (Input, MaxPooling2D, SeparableConv2D, UpSampling2D, Multiply, concatenate)
from tensorflow.keras.models import Model
from tqdm import tqdm
from numpy.fft import fft, ifft, fftshift

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
    nSlices = slices.shape[2]
    logslicesAmp = abs(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # and retrieve the phase
    logslicesPhase = np.angle(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # reescale amplitude
    logslicesAmp = np.log10(logslicesAmp)
    slicesMax = np.reshape(logslicesAmp.max(axis=(0, 1)), ( 1, 1,nSlices))
    slicesMin = np.reshape(logslicesAmp.min(axis=(0, 1)), ( 1, 1,nSlices))
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

def inverseLogScaleSummary(oldslices, slicesMax, slicesMin):
 
    slices = np.copy(oldslices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[ :, :, 0] + 1j*slices[ :, :, 1])
    slicesPhase = np.angle(slices[ :, :, 0] + 1j*slices[ :, :, 1])
    slicesAmp = slicesAmp * (slicesMax - slicesMin) + slicesMin
    slicesAmp = 10**(slicesAmp)
    slices[:, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices

def define_discriminator(image_shape):


	init = RandomNormal(stddev=0.02)
	in_src_image = Input(shape=image_shape)
	in_target_image = Input(shape=image_shape)
	
	merged = Concatenate()([in_src_image, in_target_image])
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)

	model = Model([in_src_image, in_target_image], patch_out)
	
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=0.5)
	return model

def unet_generator(input_shape):

    inputs = Input(input_shape)
    def encoder_block(x, filters):
        x = SeparableConv2D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)  
        x = Activation('gelu')(x)
        p = SeparableConv2D(filters, 2, strides=2, padding='same')(x)
        p = Activation('gelu')(p)
        return x, p
    
    def decoder_block(x, skip, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = Multiply()([skip, x])
        x = SeparableConv2D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        return x

    conv1, pool1 = encoder_block(inputs, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)
    conv5, pool5 = encoder_block(pool4, 512)

    conv6 = SeparableConv2D(1024, 7, padding='same')(pool5)
    conv6 = Activation('gelu')(conv6)

    conv7 = decoder_block(conv6, conv5, 512)
    conv8 = decoder_block(conv7, conv4, 512)
    conv9 = decoder_block(conv8, conv3, 256)
    conv10 = decoder_block(conv9, conv2, 128)
    conv11 = decoder_block(conv10, conv1, 64)
    decoded = SeparableConv2D(2, 3, activation='tanh', padding='same')(conv11)
    autoencoder = Model(inputs, decoded)
    return autoencoder

def content_loss(y_true, y_pred):
    diff = y_true - y_pred
    diff_plus_one_squared = tf.square(diff + 1) + 1
    loss = tf.reduce_mean(diff_plus_one_squared * diff)
    return loss

def adversarial_loss(y_true, y_pred):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(y_pred), y_pred) * y_true
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(y_pred), y_pred) * (1 - y_true)
    total_loss = real_loss + fake_loss
    return total_loss

# vgg = VGG19(include_top=False, weights='imagenet')
# vgg.trainable = False

# def feature_loss(y_true, y_pred):
#     if y_true.shape[-1] == 1:
#         y_true = tf.image.grayscale_to_rgb(y_true)
#         y_pred = tf.image.grayscale_to_rgb(y_pred)
    
#     true_features = vgg(y_true)
#     pred_features = vgg(y_pred)
#     loss = tf.reduce_mean(tf.square(true_features - pred_features))
#     return loss


def total_loss(y_true, y_pred, L=0.05, M=0.15, N=0.8):
    content = content_loss(y_true, y_pred)
    # feature = feature_loss(y_true, y_pred)
    adversarial = adversarial_loss(y_true, y_pred)
    total = L * content  + N * adversarial #+ M * feature
    return total

def define_gan(g_model, d_model, image_shape, L=0.05, M=0.15, N=0.8):

    for layer in d_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=lambda y_true, y_pred: total_loss(y_true, y_pred, L, M, N), optimizer=opt)
    
    return model

def generate_real_samples(dataset, n_samples, patch_shape,slicesmin,slicesmax):
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    smax = slicesmax[:,:,ix]
    smin = slicesmin[:,:,ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y,smin,smax

def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def summarize_performance(step, g_model, dataset, n_samples=3):
    # path = '/home/dapulgaris/Models/cxpix2pixcomplexdbscale2'
    path = r'E:\models\ccarnet'
    [X_realA, X_realB], _,smin,smax = generate_real_samples(dataset, n_samples,
                                                            1,combined_min,
                                                            combined_max)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    vmin = 70
    vmax = 150
    for i in range(n_samples):
        invslice0 = inverseLogScaleSummary(X_realA[i], smax[:,:,i,1], smin[:,:,i,1])
        plot0 = dbscale(invslice0)
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(plot0, cmap='hot',aspect='auto',vmin=vmin,vmax=vmax)
    # plot generated target image
    for i in range(n_samples):
        invslice1 = inverseLogScaleSummary(X_fakeB[i],smax[:,:,i,1], smin[:,:,i,1])
        plot1 = 20*np.log10(abs((invslice1[:,:,0]+1j*invslice1[:,:,1])))
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(plot1, cmap='hot',aspect='auto',vmin=vmin,vmax=vmax)
    # plot real target image
    for i in range(n_samples):
        invslice2 = inverseLogScaleSummary(X_realB[i], smax[:,:,i,0], smin[:,:,i,0])
        plot2 = dbscale(invslice2)
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(plot2, cmap='hot',aspect='auto',vmin=vmin,vmax=vmax)
    # save plot to file
    filename1 = path +'/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = path+'/model_%06d.h5' % (step+1)
    g_model.save(filename2)

def train(d_model, g_model, gan_model, dataset, n_epochs, n_batch=1):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    d_loss1_val = []
    d_loss2_val = []
    g_loss_val  = []

    for i in range(n_steps):
        [X_realA, X_realB], y_real,_,_ = generate_real_samples(dataset, n_batch, n_patch,combined_max,combined_min)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss_results = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        g_loss = g_loss_results[0]
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
dataPath = '/shared-dirs/optica/DLOCT-DATA/Experimental_Data'
noArtifacts = 'tomogram_no_artifacts'
artifacts = 'tomogram_artifacts'
folders = ['depth_nail']
tomTarget = []
tomInput = []
# for folder in folders:
#     # pathcomplex = os.path.join(dataPath,artifacts,folder)
#     # artifact_files = os.listdir(pathcomplex)
#     # for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
#     #         real_file_path = os.path.join(pathcomplex, real_file)
#     #         imag_file_path = os.path.join(pathcomplex, imag_file)
#     #         print(f'real: {real_file_path}')
#     #         print(f'imag: {imag_file_path}')
#     #         dimensions = extract_dimensions(real_file[:-4])
#     #         tomReal = read_tomogram(real_file_path, dimensions)
#     #         tomImag = read_tomogram(imag_file_path, dimensions)
#     #         tomcc = np.stack((tomReal,tomImag),axis=3)
#     #         del tomImag, tomReal


#     pathcomplex = os.path.join(dataPath,noArtifacts,folder)
#     artifact_files = os.listdir(pathcomplex)
#     for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
#             real_file_path = os.path.join(pathcomplex, real_file)
#             imag_file_path = os.path.join(pathcomplex, imag_file)
#             dimensions = extract_dimensions(real_file[:-4])
#             tomReal = read_tomogram(real_file_path, dimensions)
#             tomImag = read_tomogram(imag_file_path, dimensions)
#             tom = np.stack((tomReal,tomImag),axis=3)
#             del tomImag, tomReal

#     tomcc = tom[:,:,:,0]+1j*tom[:,:,:,1] + np.flip(tom[:,:,:,0]+1j*tom[:,:,:,1],axis=0)
#     tomcc = np.stack((tomcc.real,tomcc.imag),axis=3)
#     size = 512
#     initz = 256
#     initx1 = 0
#     initx2 = 512
#     tom1 = tom[initz:initz+size,initx1:initx1+size,:,:]
#     tom1cc = tomcc[initz:initz+size,initx1:initx1+size,:,:]
#     tom2 = tom[initz:initz+size,initx2:,:,:]
#     tom2cc = tomcc[initz:initz+size,initx2:,:,:]
#     print(tom1.shape)
#     print(tom2.shape)
#     tomTarget.append(tom1)
#     tomInput.append(tom1cc)
#     tomTarget.append(tom2)
#     tomInput.append(tom2cc)
#     print(f'tomogram {folder} loaded')
# del tom,tomcc,tom1,tom1cc,tom2,tom2cc
#%%reading sinthetic tomograms
# dataPath = '/shared-dirs/optica/DLOCT-DATA/Simulated/tom_no_artifact'
dataPath = r'E:\DLOCT\Simulated_Data_Complex\tom_no_artifact'
sinteticlist = os.listdir(dataPath)
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
        tomccsyntetic = ifft(tomsyntetic[:,:,:,0]+1j*tomsyntetic[:,:,:,1],axis=0)
        tomccsyntetic = fft(abs(tomccsyntetic)*np.cos(np.angle(tomccsyntetic)),axis=0)
        tomccsyntetic = np.stack((tomccsyntetic.real,tomccsyntetic.imag),axis=3)
        tomTarget.append(tomsyntetic)
        tomInput.append(tomccsyntetic)
del tomReal,tomImag, tomsyntetic, tomccsyntetic
#%% normalization amplitude and phase process process
tomTargetNorm = []
tomInputNorm = []
tomTargetmax =[]
tomTargetmin =[]
tomInputmax =[]
tomInputmin =[]
c = 0
for t in tqdm(range(len(tomInput))):
    tomNorm,tmax,tmin = logScale(tomTarget[t]+c)
    tomccNorm,imax,imin = logScale(tomInput[t]+c)
    tomTargetNorm.append(tomNorm)
    tomInputNorm.append(tomccNorm)
    tomTargetmax.append(tmax)
    tomTargetmin.append(tmin)
    tomInputmax.append(imax)
    tomInputmin.append(imin)
del tomTarget, tomInput, tomNorm,tomccNorm,tmax,tmin,imax,imin
print('normalized tomograms')
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
combined_input_max = np.concatenate(tomInputmax, axis=2)
combined_input_min = np.concatenate(tomInputmin, axis=2)
combined_target_max = np.concatenate(tomTargetmax, axis=2)
combined_target_min = np.concatenate(tomTargetmin, axis=2)
combined_max = np.stack((combined_target_max,combined_input_max),axis=3)
combined_min = np.stack((combined_target_min,combined_input_min),axis=3)
print(combined_target.shape)
del tomTargetNorm, tomInputNorm, tomTargetmax, tomTargetmin, tomInputmax, tomInputmin
#%%
d_loss1_epoch = []
d_loss2_epoch = []
g_loss_epoch  = []
n_steps_epoch = []
n_epochs = 2
dataset=[combined_input,combined_target]
image_shape = (combined_target.shape[1],combined_target.shape[2],2)
print('Loaded', dataset[0].shape, dataset[1].shape)
print('image shape: ',image_shape)
d_model = define_discriminator(image_shape)
g_model = unet_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)
train(d_model, g_model, gan_model,dataset,n_epochs)
# np.save('/home/dapulgaris/Models/cxpix2pixcomplexdbscale2/d_loss1', d_loss1_epoch)
# np.save('/home/dapulgaris/Models/cxpix2pixcomplexdbscale2/d_loss2', d_loss2_epoch)
# np.save('/home/dapulgaris/Models/cxpix2pixcomplexdbscale2/g_loss',  g_loss_epoch)
