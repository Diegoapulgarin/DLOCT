#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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
#%%
path = r'D:\DLOCT\ultimo experimento subsampling paper\Fovea'
file = 'tomFlatDataOver_z=400_x=896_y=960_pol1.npy'
# file2 = 'tomDataOver_z=560_x=1024_y=1024_pol2.npy'
# file2 = 'tomDataOver_z=560_x=1024_y=1024.npy'
# file3 = 'Tom_z=1152_x=1024_y=1024.npy'
# tomOver = np.load(os.path.join(path,file))
tomOver1 = np.load(os.path.join(path,file))
# tomOver2 = np.load(os.path.join(path,file2))
# tomOver = tomOver1 + 0.5*tomOver2
# tomdatas2 = np.load(os.path.join(path,file3))
# tomdatas2 = tomdatas2[400:960,:,:]
#%%
# fnameTom = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomFlat_z=400_x=896_y=960_pol=2' # fovea
# tomShape = [(400,896,960,2)]# porcine cornea
# fname = os.path.join(path, fnameTom)
# # Names of all real and imag .bin files
# fnameTomReal = [fname + '_real.bin' ]
# fnameTomImag = [fname + '_imag.bin' ]
# tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
# tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using

# tomImag = np.fromfile(fnameTomImag[0],'single')
# tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using

# tomDatas = np.stack((tomReal,tomImag), axis=4)
# del tomImag, tomReal
# tomDatas = tomDatas[:,:,:,:,0] + 1j* tomDatas[:,:,:,:,1]
#%%
# tomOver1 = tomOver1[:,:,:,0]+1j*tomOver1[:,:,:,1]
#%%
datatoprocess = tomOver1
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

#%%
tomOver1 = tomOver1[:,:,:,0]+1j*tomOver1[:,:,:,1]
#%%
save_path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\partial_results\segunda revisión\foveaflat'
dpi = 300
n = 150
plot = 10*np.log10(abs(tomOver1[:,:,n])**2)
plt.rcParams['figure.dpi']=dpi
plt.imshow(plot,cmap='gray',vmax=120,vmin=60)
plt.axis('off')
plt.savefig(os.path.join(save_path,f'nofoveaFlat_reconstructed_y={n}_{np.shape(plot)[0]}x{np.shape(plot)[1]}')
            , dpi=dpi, format=None, metadata=None,
            bbox_inches='tight', pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None,
        )
#%%
