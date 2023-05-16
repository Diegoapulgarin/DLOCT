import numpy as np
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
path = '/home/dapulgaris/data/tomDataOver_Fovea_pol'
ext = '.npy'
pol = ['1','2']
tomDataOverOf = []
for i in pol:
    print(path+i+ext)
    tomDataOver = np.load(path+i+ext)
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
    savepath = path+i+'.mat'
    tomDataOverOf = np.array(tomDataOverOf)
    mdic = {"tomDataOver": tomDataOverOf, "label": "tomDataOver"}
    savemat(savepath, mdic)
