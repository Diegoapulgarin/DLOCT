# -*- coding: utf-8 -*-
"""
Module with all metrics, quantitative and qualitative
"""
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import numpy as np

import sys
sys.path.append(r'C:\Users\labfisica\Documents\Intercambio_Informacion_EAFIT\Analysis\DLOCT\TrainingModels')

from Utils import gaussfit

#%% Define figure options

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

pio.templates.default = 'presentation'

# figsize with conversion factor from mm to inches
# (338.67, 190.5) is slide size
mm2pixels = 3.7795275591

savefolder = r'G:\Data\DLOCT\outputs\Images\QualitativeMetrics'
dpi = 300

color_list = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]

#%% Quantitative metrics

def ownPhaseMetric(slices, slicesOver): # that is (y_true, y_predicted)
    slicesPhase = tf.atan2(slices[:, :, :, 1], slices[:, :, :, 0])
    slicesOverPhase = tf.atan2(slicesOver[:, :, :, 1], slicesOver[:, :, :, 0])
    ephase = tf.subtract(slicesPhase, slicesOverPhase)
    sdphase = tf.subtract(tf.reduce_mean(tf.square(ephase), axis=[-2,-1]),
                          tf.square(tf.reduce_mean(ephase, axis=[-2,-1])))
    sdphase = tf.sqrt(sdphase)
    return sdphase

def ownPhaseMetricCorrected(slices, slicesOver):
    
    #Re-scale to -1,1
    slices = (slices * 2) - 1
    slicesOver = (slicesOver * 2) - 1
    
    slicesPhase = tf.atan2(slices[:, :, :, 1], slices[:, :, :, 0])
    slicesOverPhase = tf.atan2(slicesOver[:, :, :, 1], slicesOver[:, :, :, 0])
    ephase = tf.subtract(slicesPhase, slicesOverPhase)
    sdphase = tf.subtract(tf.reduce_mean(tf.square(ephase), axis=[-2,-1]),
                          tf.square(tf.reduce_mean(ephase, axis=[-2,-1])))
    sdphase = tf.sqrt(sdphase)
    return sdphase

def ownPhaseMetric_numpy(slices, slicesOver): # that is (y_true, y_predicted)
    slicesPhase = np.arctan2(slices[:, :, :, 1], slices[:, :, :, 0])
    slicesOverPhase = np.arctan2(slicesOver[:, :, :, 1], slicesOver[:, :, :, 0])
    ephase = slicesPhase - slicesOverPhase
    sdphase = np.mean((ephase)**2, axis=(1,2)) - (np.mean(ephase, axis=(1,2)))**2
    sdphase = np.sqrt(sdphase)
    return sdphase

def ownPhaseMetricCorrected_numpy(slices, slicesOver): # that is (y_true, y_predicted)

    #Re-scale to -1,1
    slices = (slices * 2) - 1
    slicesOver = (slicesOver * 2) - 1
    
    slicesPhase = np.arctan2(slices[:, :, :, 1], slices[:, :, :, 0])
    slicesOverPhase = np.arctan2(slicesOver[:, :, :, 1], slicesOver[:, :, :, 0])
    ephase = slicesPhase - slicesOverPhase
    sdphase = np.mean((ephase)**2, axis=(1,2)) - (np.mean(ephase, axis=(1,2)))**2
    sdphase = np.sqrt(sdphase)
    return sdphase

def ssimMetric(slices, slicesOver): # that is (y_true, y_predicted)
    slicesInt = slices[:, :, :, 0]**2 + slices[:, :, :, 1]**2
    slicesOverInt = slicesOver[:, :, :, 0]**2 + slicesOver[:, :, :, 1]**2
    ssims = []
    for i in range(len(slicesInt)):
        ssims.append(ssim(slicesInt[i, :, :], slicesOverInt[i, :, :]))
    return np.array(ssims)


def mseMetric(slices, slicesOver):
    
    mses = (slices - slicesOver)**2
    mses = np.mean(mses, axis=(1,2,3))
    
    return np.array(mses)

def MSEstatDistribution(slices, slicesOver, channel=0, normalize=False,
                        density=True):
    
    NSlices = slices.shape[0]
    MSEs = np.zeros(NSlices)
    for i in range(NSlices):
        MSEs[i] = statDistributionNormalized(slices[i, :, :, channel],
                                             slicesOver[i, :, :, channel],
                                             plotting=False,
                                             fitting=False,
                                             savename=None,
                                             normalize=normalize,
                                             density=density)
    return MSEs

def MSEPowerSpectrumComparison(slices, slicesOver, normalize=False, meandim=0, ):
    
    NSlices = slices.shape[0]
    MSEs = np.zeros(NSlices)
    for i in range(NSlices):
        MSEs[i] = PowerSpectrumComparison(slices[i], slicesOver[i],
                                              plotting=False,
                                              savename=None,
                                              normalize=normalize,
                                              meandim=meandim)
    return MSEs

#%% Qualitative metrics

def statDistributionNormalized(slices, slicesOver,
                               title='Statistical Distribution',
                               plotting=True,
                               savename=None,
                               fitting=True,
                               normalize=True,
                               density=True,
                               distrange=None):
    
    sigmaguess = np.std(slices)
    
    countSlices, binslices = np.histogram(slices, bins='auto', density=density)
    binscenters = (binslices[:-1] + binslices[1:])/2
        
    countSlicesOver, binslices = np.histogram(slicesOver, bins=binslices, density=density)
    
    if density:
        countSlices = countSlices/100
        countSlicesOver = countSlicesOver/100
    
    if normalize:
        countSlices = countSlices/np.max(countSlices)
        countSlicesOver = countSlicesOver/np.max(countSlicesOver)
        yaxisTitle= 'Counts (Normalized)'
    
    else:
        yaxisTitle = 'Probability'
    
    
    if fitting:
        histFitSlices, r_squaredSlices, p = gaussfit(
            binscenters,
            countSlices,
            [np.max(countSlices), sigmaguess]
            )
        sigmaSlices = p[1]
        
        histFitSlicesOver, r_squaredSlicesOver, p = gaussfit(
            binscenters,
            countSlicesOver,
            [np.max(countSlicesOver), sigmaguess]
            )
        sigmaSlicesOver = p[1]
    
    MSEcounts = np.mean((countSlices - countSlicesOver)**2)
    
    # plotting
    if plotting:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=binscenters, y=countSlices, mode='lines',
                                 name='Original',
                                 line=dict(color=color_list[0])
                                 ))
                
        fig.add_trace(go.Scatter(x=binscenters, y=countSlicesOver, mode='lines',
                                 name='Upsampled',
                                 line=dict(color=color_list[1])
                                 ))
        
        if fitting:
        
            fig.add_trace(go.Scatter(x=binscenters,
                                  y=histFitSlices,
                                  mode='lines',
                                  name='R^2: {:.2f}, std: {:.2e}'.format(r_squaredSlices, sigmaSlices),
                                  line=dict(dash='dash',
                                             color=color_list[0])
                                  ))
            
            fig.add_trace(go.Scatter(x=binscenters,
                                  y=histFitSlicesOver,
                                  mode='lines',
                                  name='R^2: {:.2f}, std: {:.2e}'.format(r_squaredSlicesOver, sigmaSlicesOver),
                                  line=dict(dash='dash',
                                             color=color_list[1])
                                  ))
        if distrange is not None:
            fig.update_xaxes(range=distrange,
                             tickformat='.1e')
        else:
            fig.update_xaxes(tickformat='.1e')    
        fig.update_yaxes(
            tickformat='.1e')
        fig.update_layout(
        # width=140 * mm2pixels,
        # height=140/2 * mm2pixels,
        font_family="Times New Roman",
        font_size=16,
        margin=dict(l=55, r=10, t=35, b=60),
        title=title,
        xaxis_title='Values',
        yaxis_title=yaxisTitle,
        )
        fig.show(renderer = 'svg+notebook')
        
        if savename is not None:
            fig.write_image(savename)
    
    
    return MSEcounts


def PowerSpectrumComparison(slices, slicesOver, slicesUnder, latSamp=1, cmap='viridis', plotting=True,
                      savename=None, meandim=0, PSrange=None, normalize=False):
    
    ftslice = np.fft.ifftshift(np.fft.fft2( np.fft.fftshift(slices[:, :, 0] + 1j*slices[:, :, 1])))
    ftsliceOver = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(slicesOver[:, :, 0] + 1j*slicesOver[:, :, 1])))
    ftsliceUnder = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(slicesUnder[:, :, 0] + 1j*slicesUnder[:, :, 1])))
    nx, ny = slices[:, :, 0].shape
    if meandim == 0: # I keep the dimension that I do not average
        faxis = np.linspace(-0.5, 0.5 - 1/ny, ny) * (1/latSamp)
    else:
        faxis = np.linspace(-0.5, 0.5 - 1/nx, nx) * (1/latSamp)
    
    
    if normalize:
        powerSpectrumSlice = abs(ftslice)**2 / np.max(abs(ftslice)**2)
        powerSpectrumSliceOver = abs(ftsliceOver)**2 / np.max(abs(ftsliceOver)**2)
        powerSpectrumSliceUnder = abs(ftsliceUnder)**2 / np.max(abs(ftsliceUnder)**2)
        yaxisTitle='Power Spectrum Normalized'
    else:
        powerSpectrumSlice = abs(ftslice)**2
        powerSpectrumSliceOver = abs(ftsliceOver)**2
        powerSpectrumSliceUnder = abs(ftsliceUnder)**2
        yaxisTitle='Power Spectrum'
    
    # rows, cols = powerSpectrumSlice.shape
    meanPSslice = np.mean(powerSpectrumSlice, axis=meandim)
    meanPSsliceOver = np.mean(powerSpectrumSliceOver, axis=meandim)
    meanPSsliceUnder = np.mean(powerSpectrumSliceUnder, axis=meandim)
    
    if normalize:
        meanPSslice = meanPSslice / np.max(meanPSslice)
        meanPSsliceOver = meanPSsliceOver / np.max(meanPSsliceOver)
        meanPSsliceUnder = meanPSsliceUnder / np.max(meanPSsliceUnder)
    
    MSE_PS = np.mean( (powerSpectrumSlice - powerSpectrumSliceOver)**2 )

    if plotting:
        
        fig = px.imshow(powerSpectrumSlice,
                color_continuous_scale=cmap,
                title='Slice power spectrum',
                )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=20),
            coloraxis_colorbar_x=0.83,
            font_family="Times New Roman",
            font_size=16,)
        fig.show(renderer = 'svg+notebook')
        
        
        fig2 = px.imshow(powerSpectrumSliceOver,
                color_continuous_scale=cmap,
                title='SliceOver power spectrum',
                )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=40, b=20),
            coloraxis_colorbar_x=0.83,
            font_family="Times New Roman",
            font_size=16,)
        fig2.update_xaxes(showticklabels=False)
        fig2.update_yaxes(showticklabels=False)
        fig2.show(renderer = 'svg+notebook')
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=faxis,
                                 y=meanPSslice,
                                 mode='lines',
                                 name='Slice',
                                 line=dict(color=color_list[0])
                                  ))
        fig3.add_trace(go.Scatter(x=faxis,
                                 y=meanPSsliceOver,
                                 mode='lines',
                                 name='SliceOver',
                                 line=dict(color=color_list[1])
                                  ))
        fig3.add_trace(go.Scatter(x=faxis,
                                 y=meanPSsliceUnder,
                                 mode='lines',
                                 name='Sliceunder',
                                 line=dict(color=color_list[2])
                                  ))
        if PSrange is not None:
            fig3.update_yaxes(PSrange)
        fig3.update_layout(
        # width=140 * mm2pixels,
        # height=140/2 * mm2pixels,
        font_family="Times New Roman",
        font_size=16,
        margin=dict(l=55, r=10, t=35, b=60),
        title='Mean of Power Spectrum',
        xaxis_title='',
        yaxis_title=yaxisTitle
        )
        fig3.update_xaxes(showticklabels=False)
        fig3.show(renderer = 'svg+notebook')
        
        if savename is not None:
            fig.write_image(savename + '_Power_Spectrum_Slice.svg')
            fig2.write_image(savename + '_Power_Spectrum_SliceOver.svg')
            fig3.write_image(savename + '_Power_Spectrum_Mean' + str(meandim) + '.svg')
            
    return MSE_PS


def MeanPowerSpectrumComparison(slices, slicesOver, latSamp=1, cmap='viridis', plotting=True,
                      savename=None, meandim=0, normalize=True):
    
    ftslice = np.fft.ifftshift(
        np.fft.fft2(
            np.fft.fftshift(
                slices[:, :, :, 0] + 1j*slices[:, :, :, 1], axes=(-2, -1))
            ), axes=(-2, -1)
        )
    
    ftsliceOver = np.fft.ifftshift(
        np.fft.fft2(
            np.fft.fftshift(
                slicesOver[:, :, :, 0] + 1j*slicesOver[:, :, :, 1], axes=(-2, -1))
            ), axes=(-2, -1)
        )
    
    nx, ny = slices[0, :, :, 0].shape
    if meandim == 0: # I keep the dimension that I do not average
        faxis = np.linspace(-0.5, 0.5 - 1/ny, ny) * (1/latSamp)
    else:
        faxis = np.linspace(-0.5, 0.5 - 1/nx, nx) * (1/latSamp)
    
    MeanPowerSpectrum = np.mean(abs(ftslice)**2, axis=0)
    MeanPowerSpectrumOver = np.mean(abs(ftsliceOver)**2, axis=0)
    
    yaxisTitle = 'Mean Power Spectrum [A.U]'
    if normalize:
        MeanPowerSpectrum = MeanPowerSpectrum/np.max(MeanPowerSpectrum)    
        MeanPowerSpectrumOver = MeanPowerSpectrumOver/np.max(MeanPowerSpectrumOver)
        yaxisTitle = 'Mean Power Spectrum (Normalized)'

    # rows, cols = MeanPowerSpectrum.shape
    meanMPS = np.mean(MeanPowerSpectrum, axis=meandim)
    meanMPSOver = np.mean(MeanPowerSpectrumOver, axis=meandim)
    
    if normalize:
        meanMPS = meanMPS / np.max(meanMPS)
        meanMPSOver = meanMPSOver / np.max(meanMPSOver) * 1.05
    
    MSE_MPS = np.mean( (MeanPowerSpectrum - MeanPowerSpectrumOver)**2 )

    if plotting:
        
        fig = px.imshow(MeanPowerSpectrum,
                color_continuous_scale=cmap,
                title='Ground-truth Tomogram Mean Power Spectrum',
                )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=20),
            coloraxis_colorbar_x=0.83,
            font_family="Times New Roman",
            font_size=16,)
        fig.show(renderer = 'svg+notebook')
        
        
        fig2 = px.imshow(MeanPowerSpectrumOver,
                color_continuous_scale=cmap,
                title='Reconstructed Tomogram Mean Power Spectrum',
                )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=40, b=20),
            coloraxis_colorbar_x=0.83,
            font_family="Times New Roman",
            font_size=16,)
        fig2.update_xaxes(showticklabels=False)
        fig2.update_yaxes(showticklabels=False)
        fig2.show(renderer = 'svg+notebook')
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=faxis,
                                 y=meanMPS,
                                 mode='lines',
                                 name='Ground truth',
                                 line=dict(color=color_list[0])
                                  ))
        fig3.add_trace(go.Scatter(x=faxis,
                                 y=meanMPSOver,
                                 mode='lines',
                                 name='Reconstructed',
                                 line=dict(color=color_list[1])
                                  ))
        
        fig3.update_layout(
        # width=140 * mm2pixels,
        # height=140/2 * mm2pixels,
        font_family="Times New Roman",
        font_size=16,
        margin=dict(l=55, r=10, t=35, b=60),
        title='Mean of Mean Power Spectrum',
        xaxis_title='Frequency',
        yaxis_title=yaxisTitle,
        )
        # fig3.update_xaxes(showticklabels=False)
        fig3.show(renderer = 'svg+notebook')
        
        if savename is not None:
            fig.write_image(savename + '_Mean_Power_Spectrum_Slice.svg')
            fig2.write_image(savename + '_Mean_Power_Spectrum_SliceOver.svg')
            fig3.write_image(savename + '_Mean_Power_Spectrum_Mean_dim' + str(meandim) + '.svg')
            
    return MSE_MPS

def Correlation(slices, savename=None):
    slices = slices[:,:,0] + 1j*slices[:,:,1]
    
    correlationx = np.angle(slices[:,1:] * np.conjugate(slices[:,:-1]))
    correlationy = np.angle(slices[1:, :] * np.conjugate(slices[:-1, :]))
    stdx = np.std(correlationx)
    meanx = np.mean(correlationx)
    
    stdy = np.std(correlationy)
    meany = np.mean(correlationy)
    
    fig = px.imshow(correlationx,
                color_continuous_scale='Twilight',
                title='Correlation X, mean: {:.3f}, std: {:.3f}'.format(meanx, stdx),
                zmin= -np.pi,
                zmax = np.pi
                )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=20),
        coloraxis_colorbar_x=0.83,
        font_family="Times New Roman",
        font_size=16,)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show(renderer = 'svg+notebook')
        
        
    fig2 = px.imshow(correlationy,
                color_continuous_scale='Twilight',
                title='Correlation Y, mean: {:.3f}, std: {:.3f}'.format(meany, stdy),
                zmin= -np.pi,
                zmax = np.pi
                )
    fig2.update_layout(
        margin=dict(l=10, r=10, t=40, b=20),
        coloraxis_colorbar_x=0.83,
        font_family="Times New Roman",
        font_size=16,)
    fig2.update_xaxes(showticklabels=False)
    fig2.update_yaxes(showticklabels=False)
    fig2.show(renderer = 'svg+notebook')
    
    if savename is not None:
        fig.write_image(savename + '_CorrelationX.svg')
        fig2.write_image(savename + '_CorrelationY.svg')

