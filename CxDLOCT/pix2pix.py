"""
pix2pix function

"""
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Reshape, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Conv1D, Conv1DTranspose, LeakyReLU, Dropout
from keras.initializers import RandomNormal
from keras.layers import Activation
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD


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
    g = Conv1DTranspose(image_shape[1], 4, strides=2, padding='same', kernel_initializer=init)(d6)
    out_image = Activation('relu')(g)

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
        plt.plot(X_realA[i])
        plt.title('Real Source')

    # plot generated target line plots
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.plot(X_fakeB[i])
        plt.title('Generated')

    # plot real target line plots
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        plt.plot(X_realB[i])
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