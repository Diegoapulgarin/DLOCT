# -*- coding: utf-8 -*-
"""
Module with all CNN general arquitectures
"""
#%% Import

# General libraries
# import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Customized modules
from Metrics import ownPhaseMetric, ownPhaseMetricCorrected


#%% Autoencoders

def Autoencoder2(parameters):
    """
    Asymmetric autoencoder

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    
    inputShape = parameters['inputShape']
    
    input = layers.Input(shape=inputShape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      input_shape=inputShape)(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 64,64
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 32, 32
    
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 16, 16
    
    # Decoder
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    
    x = layers.UpSampling2D((2, 2))(x)  # 32, 32
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    
    x = layers.UpSampling2D((2, 2))(x)  # 64, 64
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    
    x = layers.UpSampling2D((2, 2))(x)  # 128, 128
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    
    x = layers.Conv2D(2, (3, 3), activation="relu", padding="same")(x)
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer="RMSprop",
                  loss="mean_squared_error",
                  metrics = [
                       'MeanSquaredError',
                        ownPhaseMetric,
                        ownPhaseMetricCorrected,
                      ])
    model.summary()
    
    return model


def Autoencoder3(parameters):
    
    """
    Symmetric autoencoder

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    
    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers'] # R
    NFilters = parameters['NFilters'] # N
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']

    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    NFilters = 2 * NFilters
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        NFilters = 2 * NFilters
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x) 
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(2, (3, 3), activation=chosenFinalActivation, padding="same")(x)
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model


#%% UNets

def UNet1(parameters):
    # Includes residual connections between encoding and decoding sections
    
    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']

    residualLayers = []
    
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    NFilters = 2 * NFilters
    residualLayers.append(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        NFilters = 2 * NFilters
        residualLayers.append(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x) 
        x = layers.add([x, residualLayers[-(i+1)]])
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(2, (3, 3), activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

def UNet2(parameters):
    # Includes dropout layers for regularization
    
    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']

    residualLayers = []
    
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    NFilters = 2 * NFilters
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    residualLayers.append(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        NFilters = 2 * NFilters
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        residualLayers.append(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        x = layers.UpSampling2D((2, 2))(x) 
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        x = layers.add([x, residualLayers[-(i+1)]])
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(2, (3, 3), activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

def UNet3(parameters):
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    NFilters = 2 * NFilters
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    # x = layers.BatchNormalization()(x)
    residualLayers.append(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        NFilters = 2 * NFilters
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        residualLayers.append(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        x = layers.add([x, residualLayers[-(i+1)]])
        NFilters = NFilters / 2
    
    
    x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2DTranspose(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

def UNet4(parameters):
    # Includes batch normalization
    
    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    NFilters = 2 * NFilters
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    # x = layers.BatchNormalization()(x)
    residualLayers.append(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    x = layers.BatchNormalization()(x)
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        NFilters = 2 * NFilters
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        residualLayers.append(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        x = layers.BatchNormalization()(x)
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    x = layers.BatchNormalization()(x)

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residualLayers[-(i+1)]])
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2DTranspose(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

def UNet5(parameters):
    """
    Viene de UNet3, el cambio es que las capas finales son Conv, no ConvTranspose

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Yields
    ------
    model : TYPE
        DESCRIPTION.

    """
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    residualLayers.append(x)
    
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    NFilters = 2 * NFilters
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        residualLayers.append(x)
        
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        NFilters = 2 * NFilters
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)

        x = layers.add([x, residualLayers[-(i+1)]])
        
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model


def UNet6(parameters):
    """
    Diferencia con UNet 5 es que agrego Conv2D antes de cada Conv2dTranspose 
    del upsampling

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Yields
    ------
    model : TYPE
        DESCRIPTION.

    """
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)

    residualLayers.append(x)
    
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    NFilters = 2 * NFilters
    
    for i in range(convLayers-1):
        
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        residualLayers.append(x)
        
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        NFilters = 2 * NFilters
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    NFilters = NFilters / 2

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2D(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        x = layers.add([x, residualLayers[-(i+1)]])
        
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

def UNet7(parameters):
    """
    Diferencia con UNet 6 es que el downsampling tmb se hace con Conv2D con stride

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Yields
    ------
    model : TYPE
        DESCRIPTION.

    """
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    residualLayers.append(x)
    
    x = layers.Conv2D(NFilters, (3, 3), strides=(2,2),
                      activation=chosenActivation, padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    NFilters = 2 * NFilters
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        residualLayers.append(x)
        
        x = layers.Conv2D(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        NFilters = 2 * NFilters
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    NFilters = NFilters / 2

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2D(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        x = layers.add([x, residualLayers[-(i+1)]])
        
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model


def UNet8(parameters):
    """
    Viene de UNet5, agrego otra capa conv2d al final con kernel (3,3) con el que 
    busco corregir problemas de offset.

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Yields
    ------
    model : TYPE
        DESCRIPTION.

    """
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    residualLayers.append(x)
    
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    NFilters = 2 * NFilters
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        residualLayers.append(x)
        
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        NFilters = 2 * NFilters
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)

        x = layers.add([x, residualLayers[-(i+1)]])
        
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (3, 3), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

def UNet9(parameters):
    """
    Viene de UNet5, todos los kernels de las convolucionales se meten de (2,2)

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Yields
    ------
    model : TYPE
        DESCRIPTION.

    """
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (2, 2), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    residualLayers.append(x)
    
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    NFilters = 2 * NFilters
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (2, 2), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        residualLayers.append(x)
        
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        NFilters = 2 * NFilters
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (2, 2), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2DTranspose(NFilters, (2, 2), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)

        x = layers.add([x, residualLayers[-(i+1)]])
        
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(NFilters, (2, 2), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model


def UNet10(parameters):
    """
    Viene de UNet7, la diferencia es que los kernels de los filtros convolucionales
    NO ASOCIADOS CON UPSAMPLING O DOWNSAMPLING serán de (2,2)

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Yields
    ------
    model : TYPE
        DESCRIPTION.

    """
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (2, 2), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    residualLayers.append(x)
    
    x = layers.Conv2D(NFilters, (3, 3), strides=(2,2),
                      activation=chosenActivation, padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    NFilters = 2 * NFilters
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (2, 2), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        residualLayers.append(x)
        
        x = layers.Conv2D(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        NFilters = 2 * NFilters
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (2, 2), activation=chosenActivation, padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    NFilters = NFilters / 2

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2D(NFilters, (2, 2), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        x = layers.add([x, residualLayers[-(i+1)]])
        
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(NFilters, (2, 2), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

def UNet11(parameters):
    """
    Viene de UNet5, cambian las capas residuales por densas, excepto la última
    
    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Yields
    ------
    model : TYPE
        DESCRIPTION.

    """
    # Includes two additional Conv2DTranspose layers at the end

    # Extract parameters from dictionary
    inputShape = parameters['inputShape']
    convLayers = parameters['convLayers']
    NFilters = parameters['NFilters']
    chosenOptimizer = parameters['chosenOptimizer']
    chosenLoss = parameters['chosenLoss']
    chosenActivation = parameters['chosenActivation']
    chosenFinalActivation = parameters['chosenFinalActivation']
    dropoutRate = parameters['dropoutRate']
    
    residualLayers = []
    input = layers.Input(shape=inputShape)
    
    # Encoder
    # First conv layer is different
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same",
                      input_shape=inputShape)(input)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    residualLayers.append(x)
    
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)
    
    NFilters = 2 * NFilters
    
    for i in range(convLayers-1):
        x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        residualLayers.append(x)
        
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)
        
        NFilters = 2 * NFilters
        
    # Intermediate layer
    x = layers.Conv2D(NFilters, (3, 3), activation=chosenActivation, padding="same")(x)
    NFilters = NFilters / 2
    x = layers.Dropout(rate=dropoutRate, seed=1)(x)

    # Decoder
    for i in range(convLayers):
        x = layers.Conv2DTranspose(NFilters, (3, 3), strides=(2,2),
                                  activation=chosenActivation, padding="same")(x)
        x = layers.Dropout(rate=dropoutRate, seed=1)(x)

        x = layers.Concatenate()([x, residualLayers[-(i+1)]])
        
        NFilters = NFilters / 2
    
    # last conv is to yield same as input, that is Real and Imag components
    x = layers.Conv2D(NFilters, (3, 3), strides=(1,1),
                              activation=chosenActivation, padding="same")(x)
    
    x = layers.Conv2D(2, (1, 1), strides=(1,1),
                              activation=chosenFinalActivation, padding="same")(x)
    
    x = layers.add([x, input])
    
    # Autoencoder
    model = Model(input, x)
    model.compile(optimizer=chosenOptimizer,
                  loss=chosenLoss,
                  metrics = ['MeanSquaredError',
                               ownPhaseMetric,
                               ownPhaseMetricCorrected,
                             ])
    model.summary()
    
    return model

#%% FD-UNets