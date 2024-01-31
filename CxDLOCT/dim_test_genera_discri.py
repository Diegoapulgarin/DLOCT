#%%
import numpy as np
from keras import models,layers
import tensorflow as tf
#%%
def build_generator(input_shape=(64, 80, 2), dropout_rate=0.5):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    down1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    down1_pool = layers.MaxPooling2D((2, 2),strides=2)(down1)
    down1_dropout = layers.Dropout(dropout_rate)(down1_pool)
    
    down2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(down1_dropout)
    down2_pool = layers.MaxPooling2D((2, 2),strides=2)(down2)
    down2_dropout = layers.Dropout(dropout_rate)(down2_pool)

    down3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(down2_dropout)
    down3_pool = layers.MaxPooling2D((2, 2),strides=2)(down3)
    down3_dropout = layers.Dropout(dropout_rate)(down3_pool)

    # Decoder
    up1 = layers.UpSampling2D((2, 2))(down3_dropout)
    up1_conv = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up1)

    up2 = layers.UpSampling2D((2, 2))(up1_conv)
    up2_conv = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up2)

    up3 = layers.UpSampling2D((2, 2))(up2_conv)
    up3_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up3)

    # Salida
    outputs = layers.Conv2D(2, (1, 1), activation='tanh')(up3_conv)

    return models.Model(inputs=inputs, outputs=outputs)

from keras import layers, models

def build_discriminator(input_shape=(64, 80, 4)):
    inputs = layers.Input(shape=input_shape)

    # Capa 1
    conv1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    conv1_leaky = layers.LeakyReLU(alpha=0.2)(conv1)
    
    # Capa 2
    conv2 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(conv1_leaky)
    conv2_leaky = layers.LeakyReLU(alpha=0.2)(conv2)

    # Capa 3
    conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(conv2_leaky)
    conv3_leaky = layers.LeakyReLU(alpha=0.2)(conv3)

    # Capa de Salida
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv3_leaky)

    return models.Model(inputs=inputs, outputs=outputs)


# Suponiendo que la función build_generator ya está definida
generator = build_generator()
discriminator = build_discriminator()
# Crear datos de entrada aleatorios con la forma esperada por el generador
input_shape = (64, 80, 2)
input_shape2 = (64, 80, 4)  # Asegúrate de que esto coincida con la forma de entrada de tu generador
random_input = np.random.rand(1, *input_shape)
random_input2 = np.random.rand(1, *input_shape2)   # 1 indica un solo ejemplo

# Generar la salida usando el modelo del generador
generated_output = generator.predict(random_input)
discriminator_output = discriminator.predict(random_input2)

# Imprimir las dimensiones de la salida
print("Dimensiones de la entrada:", random_input.shape)
print("Dimensiones de la salida del generador:", generated_output.shape)

print("Dimensiones de la entrada discriminador:", random_input2.shape)
print("Dimensiones de la salida del discriminador:", discriminator_output.shape)

#%%
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

input_shape = (64, 28, 2)
random_input = np.random.rand(1, *input_shape)
generator = define_generator(input_shape)
generated_output = generator.predict(random_input)
print("Dimensiones de la entrada:", random_input.shape)
print("Dimensiones de la salida del generador:", generated_output.shape)