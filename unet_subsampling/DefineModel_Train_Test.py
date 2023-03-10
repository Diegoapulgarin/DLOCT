# -*- coding: utf-8 -*-
"""
Example on how to use the modules Metrics, Models and Utils to 
train and test a model

"""

from datetime import datetime
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# Custom modules
import sys
sys.path.append(r'C:\Users\diego\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\OCT_Advanced_Postprocessing\Analysis\DLOCT\TrainingModels')

from Utils import LoadData, logScaleSlices, downSampleSlices
from Models import UNet5

#%% """ Defining parameters """

# First, parameters to load the tomograms to use in the train

rootFolder = 'D:/DLOCT/TomogramsDataAcquisition/'


fnameTom_ = ['20-05-2022_09-04_64x256x256_25percent_4/Tomogram_1', # struct 1
            '20-05-2022_09-04_64x256x256_25percent_4/Tomogram_2', # struct 1
            '20-05-2022_09-04_64x256x256_25percent_4/Tomogram_3',# struct 1
            '20-05-2022_14-39_256x128x128_25percent_4/Tomogram_1', # struct 2
            '20-05-2022_14-39_256x128x128_25percent_4/Tomogram_2', # struct 2
            '20-05-2022_14-39_256x128x128_25percent_4/Tomogram_3', # struct 2
            '20-05-2022_14-39_256x128x128_25percent_4/Tomogram_4', # struct 2
            '20-05-2022_08-19_256x128x128_25percent_4/Tomogram_1', # struct 3
            '20-05-2022_08-19_256x128x128_25percent_4/Tomogram_2', # struct 3
            '20-05-2022_08-19_256x128x128_25percent_4/Tomogram_3', # struct 3
            '20-05-2022_08-19_256x128x128_25percent_4/Tomogram_4', # struct 3
            '18-05-2022_20-19_64x256x256_25percent_4/Tomogram_1', # struct 4
            '18-05-2022_20-19_64x256x256_25percent_4/Tomogram_2', # struct 4
            '18-05-2022_20-19_64x256x256_25percent_4/Tomogram_3', # struct 4
            '18-05-2022_20-19_64x256x256_25percent_4/Tomogram_4', # struct 4
            ]
tomStructs_ = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

# Shape of each tomogram, as tuples (Z, X, Y)
tomShape_ = [(64, 256, 256), (64, 256, 256), (64, 256, 256), (64, 256, 256),
            (256, 128, 128), (256, 128, 128), (256, 128, 128), (256, 128, 128),
            (256, 128, 128), (256, 128, 128), (256, 128, 128), (256, 128, 128),
            (64, 256, 256), (64, 256, 256), (64, 256, 256), (64, 256, 256)]

# create a dictionary with information, this are the default for the loading function
# if fnameTomData is not passed
fnameTomData = dict(fnameTom = fnameTom_,
                    tomStructs = tomStructs_,
                    tomShape = tomShape_)

slidingXSize = 128
slidingYSize = 128
strideX = slidingXSize
strideY = slidingYSize

# Now, declare some parameters for the DL model
tf.random.set_seed(1)
epochs = 300
testSize = 0.25  # Ratio of the test data wrt the total data 
batchsize = 8 # batch size of training process, increase to use full gpu

# Model parameters
inputShape = (slidingXSize, slidingYSize, 2)

# the model parameters are passed as a dictionary, for a list of the handable arguments
# check the respective function in the "Models" module

parameters=dict( 
    inputShape=inputShape,
    convLayers=4,
    NFilters=128,
    chosenOptimizer='RMSprop',
    dropoutRate=0.3,
    chosenActivation='relu',
    chosenFinalActivation='relu',
    chosenLoss='mean_squared_error',
    )

modelFunction = UNet5 # This function contains the definition of the model, 
# check all available functions in "Models" file

comment = 'UNet5_Dropout50' # comment for saving the file checkpoints and logs


#%% """ Loading data """

slices, _, _ = LoadData(rootFolder, slidingXSize, slidingYSize, strideX, strideY,fnameTomData)

logslices, slicesMax, slicesMin = logScaleSlices(slices) # transformation of data
logslicesUnder = downSampleSlices(logslices) # erasing rows and adding zeros

logX_train, logX_test, logY_train, logY_test = train_test_split(logslicesUnder,
                                            logslices,
                                            test_size=testSize,
                                            random_state=1,
                                            shuffle=True) # separating data into train and test


#%% """ Defining models """

# callback to stop early if stagnation
early = EarlyStopping(monitor='val_loss', patience=50, mode='min')

# Save checkpoints with model parameters
now = datetime.now()
logdir = 'logs/' + now.strftime('%d-%m-%Y_%H-%M-%S') + comment 
checkpoint_filepath = 'C:/Users/diego/OneDrive - Universidad EAFIT/Eafit/Trabajo de grado/' + logdir +'checkpoints/' + 'epoch_{epoch:02d}-valloss_{val_loss:.3f}'

# save folder to visualize data in tensorboard 
# (in conda, go to the folder and use >> tensorboard --logdir=\logs)
tensorboard = TensorBoard(log_dir=logdir)
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss',
                              verbose=0, save_best_only=True, mode='min')

# define model
model = modelFunction(parameters)

# Training
history = model.fit(
    x=logX_train,
    y=logY_train,
    initial_epoch=0,
    epochs=epochs,
    batch_size=batchsize,
    shuffle=True,
    validation_data=(logX_test, logY_test),
    callbacks=[tensorboard, checkpoint, early]
)
