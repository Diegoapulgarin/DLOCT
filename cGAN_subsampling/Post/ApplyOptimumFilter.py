# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:10:21 2021

@author: labfisica
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from scipy.optimize import curve_fit
import plotly.io as pio
pio.kaleido.scope.chromium_args = tuple([arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"])

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

def ApplyOptimumFilter(tomData, tomDataOver,z,vmin,vmax):
    
    tomInt = abs(tomData[z, :, :, 0] + 1j*tomData[z, :, :, 1])**2
    tomIntcom = abs(tomData[:, :, :, 0] + 1j*tomData[:, :, :, 1])**2
    tomIntDonwcomp = tomIntcom[:,1::2, :]
    tomIntDonw = tomInt[1::2, :]
    vminInt = vmin
    vmaxInt = vmax
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    dum = ax.imshow(10*np.log10(tomIntDonw), cmap='gray', vmin=vminInt, vmax=vmaxInt)
    #dum = ax.imshow((tomIntDonw), cmap='gray', vmin=vminInt, vmax=vmaxInt) provisional
    ax.set_aspect(2)
    ax.set_title('Subsampled')
    fig.colorbar(dum)
    # plt.axis('equal',adjustable='box')
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    dum = ax.imshow(10*np.log10(tomInt), cmap='gray', vmin=vminInt, vmax=vmaxInt)
    #dum = ax.imshow((tomInt), cmap='gray')#, vmin=55, vmax=75)
    ax.set_title('Original')
    fig.colorbar(dum)
    
    
    tomIntOver = abs(tomDataOver[z, :, :, 0] + 1j*tomDataOver[z, :, :, 1])**2
    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    dum = ax.imshow(10 * np.log10(tomIntOver), cmap='gray', vmin=vminInt, vmax=vmaxInt)
    #dum = ax.imshow((tomIntOver), cmap='gray', vmin=45, vmax=75)
    ax.set_title('Reconstructed')
    fig.colorbar(dum)
    
    # histograms
    countsOri, binsOri = np.histogram(10*np.log10(tomInt), bins='auto')
    binscentersOri = (binsOri[:-1] + binsOri[1:])/2
    
    countsOver, binsOver = np.histogram(10*np.log10(tomIntOver), bins=binsOri)
    
    
    countsDown, binsDown = np.histogram(10*np.log10(tomIntDonw), bins=binsOri)
    
    plt.figure()
    plt.plot(binscentersOri, countsOri, binscentersOri, countsOver, binscentersOri, countsDown)
    plt.legend(['Original', 'reconstructed', 'down'])
    plt.show()
    
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
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=np.arange(meanMPS.size),
                             y=meanMPS,
                             mode='lines',
                             name='Reconstructed',
                             line=dict(color='#1f77b4')
                              ))
    #mean power spectrum
    fig3.update_layout(
    # width=140 * mm2pixels,
    # height=140/2 * mm2pixels,
    font_family="Times New Roman",
    font_size=16,
    margin=dict(l=55, r=10, t=35, b=60),
    title='Mean of Mean Power Spectrum',
    xaxis_title='Frequency',
    yaxis_title='MPS',
    )
    # fig3.update_xaxes(showticklabels=False)
    fig3.show(renderer = 'svg+notebook')
    
    plt.figure()
    plt.imshow(MeanPowerSpectrum, cmap='viridis', vmin=None, vmax=4e11)
    plt.colorbar()
    
    
    
    meanMPS = meanMPS/np.max(meanMPS)
    
    histFitSlices, r_squaredSlices, p = gaussfit(
        np.linspace(-1, 1, meanMPS.size),
        meanMPS,
        [np.max(meanMPS), np.std(meanMPS), 0]
        )
    
    plt.figure()
    
    plt.plot(meanMPS, label='Reconstructed')
    plt.plot(histFitSlices, label='Ground-truth')
    plt.title='Mean of Mean Power Spectrum'
    plt.legend()
    plt.show()
    
    
    histFitSlicesNew = histFitSlices - (p[2] * kte )
    fit = (histFitSlicesNew - p[2]) / histFitSlicesNew
    
    fit[fit < 0] = 0
    
    fit= fit / np.max(fit)
    plt.figure(25)
    plt.plot(meanMPS, label='MPS')
    plt.plot(histFitSlices, label='Fit')
    plt.plot(fit, label='Filter')
    
    
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
      
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dum = ax.imshow(10*np.log10(abs(tomDataP[z, :, :])**2), cmap='gray', vmin=vminInt, vmax=vmaxInt)
    ax.set_title('Reconstructed with optimum filter')
    fig.colorbar(dum)
    return tomDataP, tomIntDonwcomp



