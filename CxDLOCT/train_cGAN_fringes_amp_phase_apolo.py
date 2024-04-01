#%% import libraries
import numpy as np
import os
import matplotlib.pyplot as plt 
from numpy.fft import fft, fftshift, ifft
from scipy.signal import hilbert


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
#%% functions
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
        tomogram = tomogram.reshape((depth, height, width))
    return tomogram

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
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
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
        array = X_realA[i]
        invslice0 = array[:,:,0] + 1j*array[:,:,1]
        plot0 = 10*(abs(invslice0)**2)
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(plot0, cmap='gray')
    # plot generated target image
    for i in range(n_samples):
        array = X_fakeB[i]
        invslice1 = array[:,:,0] + 1j*array[:,:,1]      
        plot1 = 10*(abs(invslice1)**2)
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(plot1, cmap='gray')
    # plot real target image
    for i in range(n_samples):
        array = X_realB[i]
        invslice2 = array[:,:,0] + 1j*array[:,:,1]
        plot2 = 10*(abs(invslice2)**2)
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(plot2, cmap='gray')
    # save plot to file
    filename1 = '/home/dapulgaris/Models/cGAN_1/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = '/home/dapulgaris/Models/cGAN_1/model_%06d.h5' % (step+1)
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

#%% read tomograms

base_path = '/shared-dirs/optica/DLOCT-DATA/Experimental_Data'

tissues = ['depth_nail','depth_chicken_breast','depth_nail_2','depth_chicken_breast2']#, 'depth_fovea', 'depth_opticNerve','depth_chicken']
all_tomograms = []
all_targets = []

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
        all_tomograms.append(tom)
        del tom, tomImag, tomReal


    for imag_file, real_file in zip(no_artifact_files[::2], no_artifact_files[1::2]):
        real_file_path = os.path.join(no_artifact_path, real_file)
        imag_file_path = os.path.join(no_artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_targets.append(tom)
    del tom, tomImag, tomReal
    print(tissue, ' loaded')

all_tomograms = np.array(all_tomograms)
all_targets = np.array(all_targets)
# all_tomograms = all_tomograms[:,:,0:10]
# all_targets = all_targets[:,:,0:10]
#%%
target_size = 512
partitions = int(dimensions[1]/target_size)
all_tomograms_partioned = []
all_targets_partioned = []
for tom in range(np.shape(all_tomograms)[0]):
    for i in range(partitions):
        xini = i*target_size
        xend = (i+1)*target_size
        minitom = all_tomograms[tom,0:target_size,xini:xend,:]
        minitarget = all_targets[tom,target_size:1024,xini:xend,:]
        all_tomograms_partioned.append(minitom)
        all_targets_partioned.append(minitarget)
del all_tomograms,all_targets
all_targets_partioned = np.transpose(np.array(all_targets_partioned),(1,2,3,0))
all_targets_partioned = np.reshape(all_targets_partioned,(target_size,target_size,(np.shape(all_targets_partioned)[3]*np.shape(all_targets_partioned)[2])))
all_tomograms_partioned = np.transpose(np.array(all_tomograms_partioned),(1,2,3,0))
all_tomograms_partioned = np.reshape(all_tomograms_partioned,(target_size,target_size,(np.shape(all_tomograms_partioned)[3]*np.shape(all_tomograms_partioned)[2])))
print('tomogram partioned')
#%%
amp_target = abs(all_targets_partioned)
phase_target = np.angle(all_targets_partioned)
amp_tomograms = abs(all_tomograms_partioned)
phase_tomograms = np.angle(all_tomograms_partioned)

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
del all_targets_partioned, all_tomograms_partioned

target_normalized = amp_target_normalized*np.exp(1j*phase_target)
tomogram_normalized = amp_tomograms_normalized*np.exp(1j*phase_tomograms)
print(np.max(target_normalized))
del amp_target_normalized, phase_target, amp_tomograms_normalized, phase_tomograms

# #revisar calculo de las franjas con matlab, revisar como es el gráfico
fringes_target = fftshift(ifft(fftshift(target_normalized,axes=0),axis=0),axes=0)
fringes_tomograms = fftshift(ifft(fftshift(tomogram_normalized,axes=0),axis=0),axes=0)
print(np.max(fringes_target))
print('fringes normalized')
# bscan = 19
shapes = np.shape(target_normalized)
X = np.zeros((shapes[0],shapes[1],shapes[2],4))
X[:, :, :, 0] = np.real(tomogram_normalized)
X[:, :, :, 1] = np.imag(tomogram_normalized)
X[:, :, :, 2] = np.real(fringes_tomograms)
X[:, :, :, 3] = np.imag(fringes_tomograms)
X = np.transpose(X,(2,0,1,3))

Y = np.zeros((shapes[0],shapes[1],shapes[2],4))
Y[:, :, :, 0] = np.real(target_normalized)
Y[:, :, :, 1] = np.imag(target_normalized)
Y[:, :, :, 2] = np.real(fringes_target)
Y[:, :, :, 3] = np.imag(fringes_target)
Y = np.transpose(X,(2,0,1,3))
print(np.shape(X))
del fringes_target,fringes_tomograms,tomogram_normalized,target_normalized
dataset = [X,Y]
shape_tom = np.shape(X)
image_shape = (shape_tom[1],shape_tom[2],4)
#%%
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
d_loss1_epoch = []
d_loss2_epoch = []
g_loss_epoch  = []
n_steps_epoch = []
n_epochs = 100
train(d_model, g_model, gan_model, dataset,n_epochs)

#%%

# #invert normalize
# fringes_original = np.zeros_like(fringes_normalized)

# for i in range(n_bscans):
#     fringes_original[:, :, i] = fringes_normalized[:, :, i] * std_vals[i] + mean_vals[i]

# fig,axs = plt.subplots(1,2)
# axs[0].imshow(((amp_target[:,:,bscan])))
# axs[0].axis('off')
# axs[0].set_title('target')
# axs[1].imshow(((amp_tomograms[:,:,bscan])))
# axs[1].axis('off')
# axs[1].set_title('artifacts')