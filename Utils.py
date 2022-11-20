# -*- coding: utf-8 -*-
"""
Module for utils, handy functions
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib



def LoadData(rootFolder, slidingXSize,
             slidingYSize,
             strideX, strideY, fnameTomdata):
    
 
    fnameTom = fnameTomdata['fnameTom']
    tomStructs = fnameTomdata['tomStructs']
    tomShape = fnameTomdata['tomShape']
        
    fnameTom = [rootFolder + fname for fname in fnameTom]
    # Names of all real and imag .bin files
    fnameTomReal = [fname + '_real.bin' for fname in fnameTom]
    fnameTomImag = [fname + '_imag.bin' for fname in fnameTom]
    
    slices = []  # Saves the fields from the original tomo, divided with sliding window
    slicesStructs = []
    tomDatas = []

    # everything happens inside the for, for each tomogram specified
    for tom in range(len(fnameTom)):
        """ Loading data """
    
        tomReal = np.fromfile(fnameTomReal[tom])
        tomReal = tomReal.reshape(tomShape[tom], order='F')  # reshape using
        # Fortran style to import according to MATLAB
    
        tomImag = np.fromfile(fnameTomImag[tom])
        tomImag = tomImag.reshape(tomShape[tom], order='F')  # reshape using
        # Fortran style to import according to MATLAB
    
        tomData = np.stack((tomReal, tomImag), axis=3)
        # tomData = tomData/np.max(abs(tomData)) # normalize tomogram
        tomDatas.append(tomData)
        
    
        """ Rearranging data : sliding window"""
    
        for z in range(tomShape[tom][0]):
            slidingYPos = 0
            # print(' z dimension :', z)
            while slidingYPos + slidingYSize <= tomShape[tom][2]:
                slidingXPos = 0
                # print('\t sliding pos y :', slidingYPos)
                while slidingXPos + slidingXSize <= tomShape[tom][1]:
                    # print('\t\t sliding pos x :', slidingXPos)
                    tomSlice = tomData[z, slidingXPos: slidingXPos + slidingXSize,
                                       slidingYPos:slidingYPos + slidingYSize, :]
                    slices.append(tomSlice)
                    slicesStructs.append(tomStructs[tom])
                    slidingXPos = slidingXPos + strideX
                slidingYPos = slidingYPos + strideY
    
    slices = np.array(slices)
    slicesStructs = np.array(slicesStructs)
     
    return slices, slicesStructs, tomDatas

def logScaleSlices(slices):
    
    logslices = np.copy(slices)
    nSlices = len(slices)
    logslicesAmp = abs(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # and retrieve the phase
    logslicesPhase = np.angle(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    # reescale amplitude
    logslicesAmp = np.log10(logslicesAmp)
    slicesMax = np.reshape(logslicesAmp.max(axis=(1, 2)), (nSlices, 1, 1))
    slicesMin = np.reshape(logslicesAmp.min(axis=(1, 2)), (nSlices, 1, 1))
    logslicesAmp = (logslicesAmp - slicesMin) / (slicesMax - slicesMin)
    # --- here, we could even normalize each slice to 0-1, keeping the original
    # --- limits to rescale after the network processes
    # and redefine the real and imaginary components with the new amplitude and
    # same phase
    logslices[:, :, :, 0] = (np.real(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    logslices[:, :, :, 1] = (np.imag(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    
    return logslices, slicesMax, slicesMin


def inverseLogScaleSlices(oldslices, slicesMax, slicesMin):
 
    slices = np.copy(oldslices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesPhase = np.angle(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesAmp = slicesAmp * (slicesMax - slicesMin) + slicesMin
    slicesAmp = 10**(slicesAmp)
    slices[:, :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    
    
    
    return slices

def downSampleSlices(slices):
    slicesUnder = np.copy(slices)
    slicesUnder[:, ::2, :, :] = 0  # this has the fields to input the model
    
    return slicesUnder

def ownPlot(tomSlice, givenTitle=None, logDisplay=False):

    if logDisplay:
        # missing to take care of 0 values
        tomSlice = np.log10(tomSlice)

    tomReal = tomSlice[:, :, 0]
    tomImag = tomSlice[:, :, 1]

    f, axs = plt.subplots(1, 2, figsize=(12, 6))
    im = axs[0].imshow(tomReal, cmap='gist_gray', vmin=0, vmax=1)
    axs[0].axis('off')
    axs[0].set_title('Real')
    im = axs[1].imshow(tomImag, cmap='gist_gray', vmin=0, vmax=1)
    axs[1].axis('off')
    axs[1].set_title('Imaginary')
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)

    if givenTitle is not None:
        plt.suptitle(givenTitle)
        
def ComparisonPlotAll_Plotly(slices, slicesOver, savename=None):
    """ Input a single field, with real and imag as 0 and 1 channels in
    3rd axis 
    
    Not working, use ComparisonPlotAll_matplotlib instead
    
    """
    
    slicesInt = abs(slices[:, :, 0] + 1j * slices[:, :, 1])**2
    slicesPhase = np.angle(slices[:, :, 0] + 1j * slices[:, :, 1])
    
    slicesOverInt = abs(slicesOver[:, :, 0] + 1j * slicesOver[:, :, 1])**2
    slicesOverPhase = np.angle(slicesOver[:, :, 0] + 1j * slicesOver[:, :, 1])
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=("Intensity", "Phase", "Real", "Imag",
                        "Intensity", "Phase", "Real", "Imag",
                        "Intensity", "Phase", "Real", "Imag"))
    
    # First row, original image
    fig_im = px.imshow(slicesInt, colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=1)
    
    fig_im = px.imshow(slicesPhase, colorscale='Twilight', zmin=-np.pi, zmax=np.pi)
    fig.add_trace(fig_im.data[0],rows = 1, cols=2)
    
    fig_im = px.imshow(slices[:, :, 0], colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=3)
    
    fig_im = px.imshow(slices[:, :, 1], colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=4)
    
    # Second row, upsampled image
    fig_im = px.imshow(slicesOverInt, colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=1)
    
    fig_im = px.imshow(slicesOverPhase, colorscale='Twilight', zmin=-np.pi, zmax=np.pi)
    fig.add_trace(fig_im.data[0],rows = 1, cols=2)
    
    fig_im = px.imshow(slicesOver[:, :, 0], colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=3)
    
    fig_im = px.imshow(slicesOver[:, :, 1], colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=4)
        
    # Third row, differences
    fig_im = px.imshow(slicesInt - slicesOverInt, colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=1)
    
    fig_im = px.imshow(slicesPhase - slicesOverPhase, colorscale='Twilight', zmin=-np.pi, zmax=np.pi)
    fig.add_trace(fig_im.data[0],rows = 1, cols=2)
    
    fig_im = px.imshow(slices[:, :, 0] - slicesOver[:, :, 0], colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=3)
    
    fig_im = px.imshow(slices[:, :, 1] - slicesOver[:, :, 1], colorscale='gray')
    fig.add_trace(fig_im.data[0], rows = 1, cols=4)
    
    fig.show(renderer='svg+notebook')
    if savename is not None:
        fig.write_image(savename + '_VisualComparison.svg')
    
def ComparisonPlotAll_matplotlib(slices, slicesOver, logslices, logslicesOver,
                                  savename=None, vminInt=None, vmaxInt=None):
    """ Input a single field, with real and imag as 0 and 1 channels in
    3rd axis 
    
    Not working, use ComparisonPlotAll_matplotlib instead
    
    """
    plt.rcParams.update({'font.size': 8})
    matplotlib.rc('pdf', fonttype=42)
    matplotlib.rc('font', family='Times New Roman') 
    # figsize with conversion factor from mm to inches
    # (338.67, 190.5) is slide size
    # mm2inches = 0.0393701
    # plt.rcParams["figure.figsize"] = (140 * mm2inches, 210*3/4 * mm2inches)
    dpi = 1200
    
    slicesInt = abs(slices[:, :, 0] + 1j * slices[:, :, 1])**2
    logsliceInt = np.log10(slicesInt)
    # logsliceInt = ( logsliceInt - np.min(logsliceInt) ) / ( np.max(logsliceInt) - np.min(logsliceInt) )
    
    if vminInt is None:
        vminInt = np.min(logsliceInt)
    if vmaxInt is None:
        vmaxInt = np.max(logsliceInt)
        
    logsliceUnder = np.copy(logsliceInt)
    logsliceUnder[::2, :] = 0
    
    slicesOverInt = abs(slicesOver[:, :, 0] + 1j * slicesOver[:, :, 1])**2
    logsliceOverInt = np.log10(slicesOverInt)
    # logsliceOverInt = ( logsliceOverInt - np.min(logsliceOverInt) ) / ( np.max(logsliceOverInt) - np.min(logsliceOverInt) )
    
    slicesPhase = np.angle(slices[:, :, 0] + 1j * slices[:, :, 1])
    slicesOverPhase = np.angle(slicesOver[:, :, 0] + 1j * slicesOver[:, :, 1])

    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)
    
    dum = axs[0, 0].imshow(logsliceInt, cmap='gist_gray', vmin=vminInt, vmax=vmaxInt)
    dum = axs[0, 1].imshow(logsliceUnder, cmap='gist_gray', vmin=vminInt, vmax=vmaxInt)
    axs[0, 2].imshow(logsliceOverInt, cmap='gist_gray', vmin=vminInt, vmax=vmaxInt),
    fig.colorbar(dum, ax= axs[0, :], label='log')
    
    dum = axs[1, 0].imshow(slicesPhase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1, 1].imshow(slicesOverPhase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1, 2].imshow(slicesPhase - slicesOverPhase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(dum, ax= axs[1, :], label='Rads')
    
    dum = axs[2, 0].imshow(logslices[:, :, 0], cmap='gist_gray', vmin=0, vmax=1)
    axs[2, 1].imshow(logslicesOver[:, :, 0], cmap='gist_gray', vmin=0, vmax=1)
    axs[2, 2].imshow(abs(logslices[:, :, 0] - logslicesOver[:, :, 0]), cmap='gist_gray', vmin=0, vmax=1)
    fig.colorbar(dum, ax= axs[2, :], label='logscale')
    
    dum = axs[3, 0].imshow(logslices[:, :, 1], cmap='gist_gray', vmin=0, vmax=1)
    axs[3, 1].imshow(logslicesOver[:, :, 1], cmap='gist_gray', vmin=0, vmax=1)
    axs[3, 2].imshow(abs(logslices[:, :, 1] - logslicesOver[:, :, 1]), cmap='gist_gray', vmin=0, vmax=1)
    fig.colorbar(dum, ax= axs[3, :], label='logscale')
    
    axs[0, 1].set_title('Intensity')
    axs[1, 1].set_title('Phase')
    axs[2, 1].set_title('Real')
    axs[3, 1].set_title('Imaginary')
    
    for i in range(4):
        for j in range(3):
            axs[i,j].axis('image')
            axs[i,j].axis('off')
          
    if savename is not None:
        fig.savefig(savename + '_VisualComparison.pdf', dpi=dpi)
        
        
def ComparisonPlotField_matplotlib(slices, slicesOver,
                                  savename=None, vminInt=None, vmaxInt=None):
    """ Input a single field, with real and imag as 0 and 1 channels in
    3rd axis 
    
    
    """
    plt.rcParams.update({'font.size': 8})
    matplotlib.rc('pdf', fonttype=42)
    matplotlib.rc('font', family='Times New Roman') 
    # figsize with conversion factor from mm to inches
    # (338.67, 190.5) is slide size
    # mm2inches = 0.0393701
    # plt.rcParams["figure.figsize"] = (140 * mm2inches, 210*3/8 * mm2inches)
    dpi = 1200
    
    slicesInt = abs(slices[:, :, 0] + 1j * slices[:, :, 1])**2
    logsliceInt = 10*np.log10(slicesInt)
    # logsliceInt = ( logsliceInt - np.min(logsliceInt) ) / ( np.max(logsliceInt) - np.min(logsliceInt) )
    
    if vminInt is None:
        vminInt = np.min(logsliceInt)
    if vmaxInt is None:
        vmaxInt = np.max(logsliceInt)
        
    logsliceUnder = np.copy(logsliceInt)
    logsliceUnder[::2, :] = 0
    
    slicesOverInt = abs(slicesOver[:, :, 0] + 1j * slicesOver[:, :, 1])**2
    logsliceOverInt = 10*np.log10(slicesOverInt)
    # logsliceOverInt = ( logsliceOverInt - np.min(logsliceOverInt) ) / ( np.max(logsliceOverInt) - np.min(logsliceOverInt) )
    
    slicesPhase = np.angle(slices[:, :, 0] + 1j * slices[:, :, 1])
    slicesOverPhase = np.angle(slicesOver[:, :, 0] + 1j * slicesOver[:, :, 1])

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    
    dum = axs[0, 0].imshow(logsliceInt, cmap='gist_gray', vmin=vminInt, vmax=vmaxInt)
    dum = axs[0, 1].imshow(logsliceUnder, cmap='gist_gray', vmin=vminInt, vmax=vmaxInt)
    axs[0, 2].imshow(logsliceOverInt, cmap='gist_gray', vmin=vminInt, vmax=vmaxInt),
    fig.colorbar(dum, ax= axs[0, :], label='Intensidad [dB]')
    
    dum = axs[1, 0].imshow(slicesPhase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1, 1].imshow(slicesOverPhase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1, 2].imshow(slicesPhase - slicesOverPhase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(dum, ax= axs[1, :], label='Rads')
    
    axs[0, 1].set_title('Intensity')
    axs[1, 1].set_title('Phase')
 

    for i in range(2):
        for j in range(3):
            axs[i,j].axis('image')
            axs[i,j].axis('off')
          
    if savename is not None:
        fig.savefig(savename + '_VisualComparison.pdf', dpi=dpi)
        
    
def gauss(x, A, sigma):
    return A * np.exp(-(x)**2/(2*sigma**2))


def gaussfit(binscenters, counts, p0):
    
    popt, pcov = curve_fit(gauss, binscenters, counts, p0, bounds=((0, 0), (np.inf, np.inf)))
    residuals = counts - gauss(binscenters, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((counts - np.mean(counts))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    hist_fit = gauss(binscenters, *popt)
    return hist_fit, r_squared, popt