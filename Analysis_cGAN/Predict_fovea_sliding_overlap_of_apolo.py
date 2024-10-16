# %%

# Custom imports

from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Metrics import ownPhaseMetric, ownPhaseMetricCorrected
from Deep_Utils import sliding_window,inv_sliding_window
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

path = '/home/dapulgaris/Models' #apolo

customObjects = {'ownPhaseMetric': ownPhaseMetric,
                 'ownPhaseMetricCorrected': ownPhaseMetricCorrected}
model = tf.keras.models.load_model(path+'/model_081920.h5',custom_objects=customObjects)

# %%
""" Load tomograms"""
rootFolder = '/home/dapulgaris/Data/' # apolo
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

tomDatas = np.stack((tomReal,tomImag), axis=5)
del tomImag, tomReal
tomDatas = np.sum(tomDatas,axis=3) # Z,X,Y,pol1-2,imag-real

num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros), (0, 0), (0, 0))
tomDatas = np.pad(tomDatas, pad_width, mode='edge')
#tomDatas = tomDatas[0:5,:,:,:,:]
#pol = [0,1]
pol = [1]
tomDataOverOf = []
# %%
for c in pol: 
    print(c)
    tomData = tomDatas[:,:,:,c,:]
    tomShape = np.shape(tomData)
    slidingYSize = 128
    slidingXSize = 128
    strideY = 128
    strideX = 128
    window_size = (slidingXSize,slidingYSize)
    step_size = (strideX,strideY)

    slices = []
    print(len(tomData))
    for i in range(len(tomData)):
        bslicei = sliding_window(tomData[i,:,:,0],window_size,step_size)
        bslicer = sliding_window(tomData[i,:,:,1],window_size,step_size)
        bslice = np.stack((bslicer,bslicei),axis=3)
        slices.append(bslice)
    print(np.shape(slices))
    slices = np.array(slices)
    slices = np.reshape(slices,(slices.shape[0]*slices.shape[1],slices.shape[2],slices.shape[3],slices.shape[4]))
    del tomData

    print(np.shape(slices))
    logslices, slicesMax, slicesMin = logScaleSlices(slices)
    del slices
    logslicesUnder = downSampleSlices(logslices)
    del logslices
    logslicesOver = np.array(model.predict(logslicesUnder, batch_size=8), dtype='float32')
    del logslicesUnder
    slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
    del logslicesOver 

    original_size = (tomDatas.shape[1],tomDatas.shape[2])
    original_planes = tomDatas.shape[0]
    origslicesOver = np.reshape(slicesOver,(original_planes,int(slicesOver.shape[0]/original_planes),slicesOver.shape[1],slicesOver.shape[2],2))
    number_planes =  origslicesOver.shape[0]
    tomDataOver = []
    for b in range(number_planes):
        bslicei,_,_ = inv_sliding_window(origslicesOver[b,:,:,:,1],window_size,original_size,step_size)
        bslicer,_,_ = inv_sliding_window(origslicesOver[b,:,:,:,0],window_size,original_size,step_size)
        bslice = np.stack((bslicer,bslicei),axis=2)
        tomDataOver.append(bslice)
    tomDataOver = np.array(tomDataOver)
    tomDataOver = tomDataOver[:, :, :tomDataOver.shape[2]-num_zeros,:]
    datatoprocess = tomDataOver

    del tomDataOver

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

    del ftslice

    fftomograma = np.fft.ifftshift(
        np.fft.fft2(
            np.fft.fftshift(
                datatoprocess[:, :, :, 0] + 1j*datatoprocess[:, :, :, 1], axes=(dim))
            ), axes=(dim)
        )

    del datatoprocess

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
    #tomDataOverOf.append(tomDataP)

    tomDataOverOf = abs(tomDataP)**2
    del tomDataP
    savepath = rootFolder + 'tomDataOverpol'+str(c+1)+'Overlap.mat'
    print(np.shape(tomDataOverOf))
    savemat(savepath, {'tomDataOver': tomDataOverOf.astype(np.float64)})
    del tomDataOverOf