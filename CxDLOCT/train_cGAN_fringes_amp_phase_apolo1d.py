#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
from numpy.fft import fft,fftshift, ifft

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



#%%
path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Data Boston\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'
file = '[p.Calibration][s.Mirror][02-10-2023_15-17-52].dispersion'
dispersion = np.fromfile(os.path.join(path,file))
# plt.plot(dispersion)
path = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscan'
artifact_files = os.listdir(path)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(path, real_file)
        imag_file_path = os.path.join(path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        del tomImag, tomReal
fringescc = fftshift(ifft(tom,axis=0),axes=0)
fringescc = fringescc[:,:,0:4]


pathtarget = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscanNoartifacts'
artifact_files = os.listdir(pathtarget)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(pathtarget, real_file)
        imag_file_path = os.path.join(pathtarget, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        del tomImag, tomReal
fringesreal = fftshift(ifft(tom,axis=0),axes=0)
fringesreal = fringesreal[:,:,0:4]
plt.imshow(20*np.log10(abs(tom[:,:,0])),cmap='gray')
#%%
# Calcular el número de puntos en el dominio k
num_puntos_k = len(dispersion)
# Crear un nuevo array k para el doble de puntos
k_nuevo = np.linspace(-1, 1, 2 * num_puntos_k)
# Interpolar la dispersión al nuevo dominio k
dispersion_interpolada = np.interp(k_nuevo, np.linspace(-1, 1, num_puntos_k), dispersion)
fftDispersion = fft(dispersion_interpolada)[0:int(len(dispersion_interpolada)/2)]
n = len(fftDispersion)
phasedispersion = np.angle(ifft(fftDispersion))
unwrapped_phase = np.unwrap(phasedispersion)
coeficientes = np.polyfit(np.arange(len(unwrapped_phase)), unwrapped_phase, 3)
polinomio = np.poly1d(coeficientes)
fase_lineal = polinomio(np.arange(len(unwrapped_phase)))
dispersivePhase = unwrapped_phase - fase_lineal
negative_phase_correction = np.exp(-1j * dispersivePhase)[:, np.newaxis, np.newaxis]
c1 = ifft(fringescc * negative_phase_correction, axis=0)
c2 = ifft(fringesreal, axis=0)
#%%
amp_target = abs(c2)
phase_target = np.angle(c2)
amp_tomograms = abs(c1)
phase_tomograms = np.angle(c1)

logamptarget = np.log(amp_target)
logamptom = np.log(amp_tomograms)
del amp_target,amp_tomograms

n_bscans = logamptarget.shape[2]
mean_vals_target = np.zeros(n_bscans)
std_vals_target = np.zeros(n_bscans)
mean_vals = np.zeros(n_bscans)
std_vals = np.zeros(n_bscans)
amp_tomograms_normalized = np.zeros_like(logamptom)
amp_target_normalized = np.zeros_like(logamptarget)
for i in range(n_bscans):
    mean_vals_target[i] = np.mean(logamptarget[:, :, i])
    std_vals_target[i] = np.std(logamptarget[:, :, i])
    mean_vals[i] = np.mean(logamptom[:, :, i])
    std_vals[i] = np.std(logamptom[:, :, i]) 
    amp_target_normalized[:, :, i] = (logamptarget[:, :, i] - mean_vals_target[i]) / std_vals_target[i]
    amp_tomograms_normalized[:, :, i] = (logamptom[:, :, i] - mean_vals[i]) / std_vals[i]

target_normalized = amp_target_normalized*np.exp(1j*phase_target)
tomogram_normalized = amp_tomograms_normalized*np.exp(1j*phase_tomograms)
del amp_target_normalized, phase_target, amp_tomograms_normalized, phase_tomograms
#%%
shapes = np.shape(target_normalized)
X = np.zeros((shapes[0],shapes[1],shapes[2],2))
X[:, :, :, 0] = np.real(tomogram_normalized)
X[:, :, :, 1] = np.imag(tomogram_normalized)
X = np.transpose(X,(2,0,1,3))
Y = np.zeros((shapes[0],shapes[1],shapes[2],2))
Y[:, :, :, 0] = np.real(target_normalized)
Y[:, :, :, 1] = np.imag(target_normalized)
Y = np.transpose(Y,(2,0,1,3))
print(np.shape(X))
print(np.shape(Y))
y, z, x, n = X.shape
X = np.reshape(X,(x*y,z,n))
Y = np.reshape(Y,(x*y,z,n))
dataset = [X,Y]
shape_tom = np.shape(X)
image_shape = (shape_tom[1],shape_tom[2])
#%%
from os import sep
import sys
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint

from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras import Input
from keras.layers import Conv1D
from keras.layers import Conv1DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import ReLU
import tensorflow as tf

from matplotlib import pyplot
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
def energy_based_symmetry_loss(y_true, y_pred, energy_threshold=1e-3):
    y_pred_complex = tf.complex(y_pred[:, :, 0], y_pred[:, :, 1])
    energy_left_half = tf.reduce_sum(tf.square(tf.abs(y_pred_complex[:, :y_pred.shape[1] // 2])), axis=1)
    energy_right_half = tf.reduce_sum(tf.square(tf.abs(y_pred_complex[:, y_pred.shape[1] // 2:])), axis=1)
    energy_difference = tf.abs(energy_left_half - energy_right_half)
    energy_excess = tf.maximum(0.0, energy_difference - energy_threshold)
    return tf.reduce_mean(energy_excess)


def phase_consistency_loss(y_true, y_pred):
    # Recomponer el espectrograma complejo
    y_true_complex = tf.complex(y_true[:, :, 0], y_true[:, :, 1])
    y_pred_complex = tf.complex(y_pred[:, :, 0], y_pred[:, :, 1])
    
    # ISTFT utilizando tf.signal.inverse_stft
    y_true_time = tf.signal.inverse_stft(y_true_complex, frame_length, frame_step)
    y_pred_time = tf.signal.inverse_stft(y_pred_complex, frame_length, frame_step)
    
    # Extraer y comparar fases
    y_true_phase = tf.math.angle(y_true_time)
    y_pred_phase = tf.math.angle(y_pred_time)
    
    # Calcular la pérdida de consistencia de fase
    phase_loss = tf.reduce_mean(tf.abs(tf.math.squared_difference(y_true_phase, y_pred_phase)))
    
    return phase_loss

def combined_loss(weights):
    # Asumiendo que 'weights' es un diccionario con las claves 'mae', 'mse', 'phase', 'energy'
    def loss(y_true, y_pred):
        # Calcular las pérdidas individuales
        mae_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        phase_loss = phase_consistency_loss(y_true, y_pred)  # Asegúrate de definir esta función
        energy_loss = energy_based_symmetry_loss(y_true, y_pred)
        
        # Combina las pérdidas con los pesos dados
        combined = (weights['mse'] * mse_loss +
                    weights['mae'] * mae_loss +
                    weights['phase'] * phase_loss +
                    weights['energy'] * energy_loss)
        return combined
    return loss

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv1D(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
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
    g = Conv1DTranspose(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
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

def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)

    # bottleneck
    b = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(e6)
    b = Activation('relu')(b)

    # decoder model
    d1 = decoder_block(b, e6, 512)
    d2 = decoder_block(d1, e5, 512)
    d3 = decoder_block(d2, e4, 512)
    d4 = decoder_block(d3, e3, 256, dropout=False)
    d5 = decoder_block(d4, e2, 128, dropout=False)
    d6 = decoder_block(d5, e1, 64, dropout=False)
    
    # output
    g = Conv1DTranspose(image_shape[1], 2, strides=2, padding='same', kernel_initializer=init)(d6)
    out_image = Activation('tanh')(g)

    # define model
    model = Model(in_image, out_image)
    return model

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
    d = Conv1D(64, 4, strides=2, padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    # C128
    d = Conv1D(128, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256
    d = Conv1D(256, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C512
    d = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C512
    d = Conv1D(512, 4, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # patch output
    d = Conv1D(1, 4, padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # slice max and min
    # smax = slicesmax[ix]
    # smin = slicesmin[ix]
    # generate âœ¬realâœ¬ class labels (1)
    y = ones((n_samples, patch_shape, 1))
    return [X1, X2], y 
    
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    
    y = zeros((len(X), patch_shape, 1))
    return X, y

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


def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    # plot real source line plots
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.plot(abs(X_realA[i][:,0]+1j*X_realA[i][:,1]))
        plt.title('Real Source')

    # plot generated target line plots
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.plot(abs(X_fakeB[i][:,0]+1j*X_fakeB[i][:,1]))
        plt.title('Generated')

    # plot real target line plots
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        plt.plot(abs(X_realB[i][:,0]+1j*X_realB[i][:,1]))
        plt.title('Real Target')

    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()

    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

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

# define the models
signal_shape = (image_shape)
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)


#%%
dataset=[X,Y]

#%%
# train model
d_loss1_epoch = []
d_loss2_epoch = []
g_loss_epoch  = []
n_steps_epoch = []
n_epochs = 25
train(d_model, g_model, gan_model, dataset,n_epochs)