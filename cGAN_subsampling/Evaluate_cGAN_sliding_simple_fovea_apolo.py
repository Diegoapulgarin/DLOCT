# %%

# Custom imports

from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Metrics import ownPhaseMetric, ownPhaseMetricCorrected
from Deep_Utils import simple_sliding_window, simple_inv_sliding_window
import numpy as np
import tensorflow as tf
# %%

#path = '/home/dapulgaris/models' #apolo
path = '/home/dapulgaris/models' #own pc
customObjects = {'ownPhaseMetric': ownPhaseMetric,
                 'ownPhaseMetricCorrected': ownPhaseMetricCorrected}
model = tf.keras.models.load_model(path+'\\Models\\model_081920.h5',custom_objects=customObjects)

# %%
""" Load tomograms"""
rootFolder = '/home/dapulgaris/Data/' # apolo
#rootFolder = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected' # own pc
fnameTom = '//[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)' # fovea
tomShape = [(586,896,960,2,2)]# porcine cornea
# %%
name = 'Experimental'
fname = rootFolder + fnameTom
# Names of all real and imag .bin files
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]

# %%
tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using

tomImag = np.fromfile(fnameTomImag[0],'single')
tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using

tomData = np.stack((tomReal,tomImag), axis=5)
del tomImag, tomReal
tomData = np.sum(tomData,axis=3) # Z,X,Y,pol1-2,imag-real

num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros), (0, 0), (0, 0))
tomData = np.pad(tomData, pad_width, mode='constant', constant_values=1)
pol = 0
tomData = tomData[:,:,:,pol,:]

# %%

tomShape = np.shape(tomData)
slidingYSize = 128
slidingXSize = 128
strideY = 128
strideX = 128
slices = simple_sliding_window(tomData,tomShape,slidingYSize,slidingXSize,strideY,strideX)

# %%
logslices, slicesMax, slicesMin = logScaleSlices(slices)
logslicesUnder = downSampleSlices(logslices)
logslicesOver = np.array(model.predict(logslicesUnder, batch_size=8), dtype='float32')
slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
tomDataOver = simple_inv_sliding_window(slicesOver, tomShape, slidingYSize, slidingXSize, strideY, strideX)
tomDataOver = tomDataOver[:, :, :tomDataOver.shape[2]-num_zeros,:]
np.save('/home/dapulgaris/data/tomDataOver.npy',tomDataOver)