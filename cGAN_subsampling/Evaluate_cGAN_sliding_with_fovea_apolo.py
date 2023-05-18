# %%

# Custom imports

from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Metrics import ownPhaseMetric, ownPhaseMetricCorrected
from Deep_Utils import sliding_window, inv_sliding_window
import numpy as np
import tensorflow as tf
# %%

#path = '/home/dapulgaris/models' #apolo
path = '/home/dapulgaris/Models' #own pc
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
tomData = np.sum(tomData,axis=3) # Z,X,Y,pol1-2,imag-real
del tomImag, tomReal
# %%
pol = 0
n = 128
s = 128
window_size = (n,n)
stride = (s,s)
slices = []
for b in range(len(tomData)):
    i = b 
    bslicei = sliding_window(tomData[i,:,:,pol,1],window_size,stride)
    bslicer = sliding_window(tomData[i,:,:,pol,0],window_size,stride)
    bslice = np.stack((bslicer,bslicei),axis=3)
    slices.append(bslice)
slices = np.array(slices)
slices = np.reshape(slices,(slices.shape[0]*slices.shape[1],slices.shape[2],slices.shape[3],slices.shape[4]))
tomData = np.stack((tomReal[:,:,:], tomImag[:,:,:]), axis=3)
del bslicei,bslicer,bslice

# %%
logslices, slicesMax, slicesMin = logScaleSlices(slices)
logslicesUnder = downSampleSlices(logslices)
logslicesOver = np.array(model.predict(logslicesUnder, batch_size=8), dtype='float32')
slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
slicesUnder=downSampleSlices(slices)
#%%
original_size = (tomShape[0][1],tomShape[0][2])
original_planes = tomData.shape[0]
origslicesOver = np.reshape(slicesOver,(original_planes,int(slices.shape[0]/original_planes),slices.shape[1],slices.shape[2],2),)
number_planes =  origslicesOver.shape[0]
tomDataOver = []
for b in range(number_planes):
    bslicei,_,_ = inv_sliding_window(origslicesOver[b,:,:,:,1],window_size,original_size,stride)
    bslicer,_,_ = inv_sliding_window(origslicesOver[b,:,:,:,0],window_size,original_size,stride)
    bslice = np.stack((bslicer,bslicei),axis=2)
    tomDataOver.append(bslice)
tomDataOver = np.array(tomDataOver)
del bslicei,bslicer,bslice

np.save('/home/dapulgaris/Data/tomDataOver.npy',tomDataOver)