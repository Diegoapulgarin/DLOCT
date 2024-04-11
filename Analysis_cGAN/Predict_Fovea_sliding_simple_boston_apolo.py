# %%

# Custom imports
import os
from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Metrics import ownPhaseMetric, ownPhaseMetricCorrected
from Deep_Utils import simple_sliding_window, simple_inv_sliding_window
import numpy as np
import tensorflow as tf
from scipy.io import savemat

from scipy.optimize import curve_fit

def gauss(x, A, sigma, offtset):
    return A * np.exp(-(x)**2/(2*sigma**2)) + offtset


def gaussfit(binscenters, counts, p0):
    
    popt, pcov = curve_fit(gauss, binscenters, counts, p0)
    residuals = counts - gauss(binscenters, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((counts - np.mean(counts))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    hist_fit = gauss(binscenters, *popt)
    return hist_fit, r_squared, popt
# %%

# path = '/home/dapulgaris/Models' #apolo
path = r'D:\DLOCT\ultimo experimento subsampling paper\Fovea' #apolo

customObjects = {'ownPhaseMetric': ownPhaseMetric,
                 'ownPhaseMetricCorrected': ownPhaseMetricCorrected}
model = tf.keras.models.load_model(path+'/model_125952.h5',custom_objects=customObjects)#cGAN
#model = tf.keras.models.load_model(path+'/BestUNetplus_DR30',custom_objects=customObjects)


# %%
""" Load tomograms"""
# rootFolder = '/home/dapulgaris/Data/DepthWrap' # apolo
rootFolder = r'D:\DLOCT\TomogramsDataAcquisition\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]' # apolo
fnameTom = '//Tom_z=1152_x=1024_y=1024' # fovea
tomShape = [(1152,1024,1024)]# porcine cornea
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

tomDatas = np.stack((tomReal,tomImag), axis=3)
del tomImag, tomReal
#tomDatas = np.sum(tomDatas,axis=3) # Z,X,Y,pol1-2,imag-real

pol = [512,1024]
tomDataOverOf = []
# %%
#%%
for i in pol: 
    print(i)
    tomData = tomDatas[200:202,:,0:i,:]
    tomDown = tomData[:,:,1::2,:]
    tomDown = tomDown[:,:,:,0] + 1j*tomDown[:,:,:,1]
    tomShape = np.shape(tomData)
    print(tomShape)
    slidingYSize = 128
    slidingXSize = 128
    strideY = 128
    strideX = 128
    slices = simple_sliding_window(tomData,tomShape,slidingYSize,slidingXSize,strideY,strideX)
    print(np.shape(slices))
    logslices, slicesMax, slicesMin = logScaleSlices(slices)
    logslicesUnder = downSampleSlices(logslices)
    logslicesOver = np.array(model.predict(logslicesUnder, batch_size=4), dtype='float32')
    slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
    tomDataOver = simple_inv_sliding_window(slicesOver, tomShape, slidingYSize, slidingXSize, strideY, strideX)
    # tomDataOver = tomDataOver[:, :, :tomDataOver.shape[2]-num_zeros,:]
    datatoprocess = tomDataOver
    dim = 1
    kte = 0.0
    ftslice = np.fft.ifftshift(
        np.fft.fft2(
            np.fft.fftshift(
                datatoprocess[:, :, :, 0] + 1j*datatoprocess[:, :, :, 1], axes=(-2, -1))
            ), axes=(-2, -1)
        )
    MeanPowerSpectrum = np.mean(abs(ftslice)**2, axis=0)
    meanMPS = np.mean(MeanPowerSpectrum, axis=1)
    meanMPS = meanMPS/np.max(meanMPS)
    histFitSlices, r_squaredSlices, p = gaussfit(
        np.linspace(-1, 1, meanMPS.size),
        meanMPS,
        [np.max(meanMPS), np.std(meanMPS), 0]
        )
    histFitSlicesNew = histFitSlices - (p[2] * kte )
    fit = (histFitSlicesNew - p[2]) / histFitSlicesNew
    fit[fit < 0] = 0
    fit= fit / np.max(fit)
    fftomograma = np.fft.ifftshift(
        np.fft.fft2(
            np.fft.fftshift(
                datatoprocess[:, :, :, 0] + 1j*datatoprocess[:, :, :, 1], axes=(dim))
            ), axes=(dim)
        )
    fit.shape
    shapefit = [1,1,1]
    shapefit[dim] = fit.size
    fit = np.reshape(fit, shapefit)
    fit.shape
    fftomograma = fftomograma * fit
    fftomograma.shape
    tomDataP = np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.fftshift(
                fftomograma[:, :, :] + 1j*fftomograma[:, :, :], axes=(dim))
            ), axes=(dim)
        )
    tomDataOverOf = tomDataP
    tomDataOverOf = tomDataP
    savepath = rootFolder + f'/tomDataOver_z={np.shape(tomDataOver)[0]}_x={np.shape(tomDataOver)[1]}_y={np.shape(tomDataOver)[2]}.npy'
    print('tomover:',np.shape(tomDataOverOf)[2])
    np.save(savepath,tomDataOverOf)
    # savemat(savepath, {'tomDataOver': tomDataOverOf.astype(np.complex32)})
    savepath = rootFolder + f'/tomDataSub_z={np.shape(tomDown)[0]}_x={np.shape(tomDown)[1]}_y={np.shape(tomDown)[2]}.npy'
    print('tomsub:',np.shape(tomDown)[2])
    np.save(savepath,tomDown)
    # savemat(savepath, {'tomDataSub': tomDown.astype(np.complex32)})
    print('######################################################')

#%%
a = np.load(savepath)