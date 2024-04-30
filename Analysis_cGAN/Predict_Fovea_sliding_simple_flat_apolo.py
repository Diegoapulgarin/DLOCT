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

path = r'D:\DLOCT\ultimo experimento subsampling paper\Fovea' #apolo

customObjects = {'ownPhaseMetric': ownPhaseMetric,
                 'ownPhaseMetricCorrected': ownPhaseMetricCorrected}
model = tf.keras.models.load_model(os.path.join(path,'model_125952.h5'),custom_objects=customObjects)#cGAN
#model = tf.keras.models.load_model(path+'/BestUNetplus_DR30',custom_objects=customObjects)
""" Load tomograms"""
rootFolder = r'D:\DLOCT\ultimo experimento subsampling paper\Fovea' # apolo
fnameTom = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomFlat_z=400_x=896_y=960_pol=2' # fovea
tomShape = [(400,896,960,2)]# porcine cornea
fname = os.path.join(rootFolder, fnameTom)
# Names of all real and imag .bin files
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]
tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using

tomImag = np.fromfile(fnameTomImag[0],'single')
tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using

tomDatas = np.stack((tomReal,tomImag), axis=4)
del tomImag, tomReal
#tomDatas = np.sum(tomDatas,axis=3) # Z,X,Y,pol1-2,imag-real
num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros), (0, 0), (0, 0))
tomDatas = np.pad(tomDatas, pad_width, mode='edge')
pol = [0,1]
dimtom = [512,1024]
tomDataOverOf = []
#%%
import matplotlib.pyplot as plt
n = 10
tomData = tomDatas[:,:,0:512,0,0]+1j* tomDatas[:,:,0:512,0,1]
plot = 10*np.log10(abs(tomData[:,:,n])**2)
plt.imshow(plot,cmap='gray',vmax=120,vmin=60)
# %%
for i in pol:
    for j in dimtom:
        print(i)
        tomData = tomDatas[:,:,0:j,i,:]
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
        logslicesOver = np.array(model.predict(logslicesUnder, batch_size=2))
        print('reconstructed pol',int(i+1))
        slicesOver = inverseLogScaleSlices((logslicesOver), slicesMax, slicesMin)
        tomDataOver = simple_inv_sliding_window(slicesOver, tomShape, slidingYSize, slidingXSize, strideY, strideX)
        if j == 512:
            tomDataOver = tomDataOver[:, :, :tomDataOver.shape[2]-int(num_zeros/2),:]
        else:
            tomDataOver = tomDataOver[:, :, :tomDataOver.shape[2]-num_zeros,:]
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
        savepath = rootFolder + f'/tomFlatDataOver_z={np.shape(tomDataOver)[0]}_x={np.shape(tomDataOver)[1]}_y={np.shape(tomDataOver)[2]}_pol{i+1}.npy'
        print('tomover:',np.shape(tomDataOverOf)[2])
        np.save(savepath,tomDataOverOf)
        # savemat(savepath, {'tomDataOver': tomDataOverOf.astype(np.complex32)})
        savepath = rootFolder + f'/tomFlatDataSub_z={np.shape(tomDown)[0]}_x={np.shape(tomDown)[1]}_y={np.shape(tomDown)[2]}_pol{i+1}.npy'
        print('tomsub:',np.shape(tomDown)[2])
        np.save(savepath,tomDown)
        # savemat(savepath, {'tomDataSub': tomDown.astype(np.complex32)})
        print('######################################################')
