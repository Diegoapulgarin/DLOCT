# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 09:04:18 2023

@author: Diego p.
"""
#%%
from os.path import join

# Custom imports
import sys
sys.path.append(r'C:\Users\diego\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\OCT_Advanced_Postprocessing_copia\Analysis\DLOCT\TrainingModels')

#from Utils import LoadData, logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Utils import LoadData, logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Utils import ComparisonPlotField_matplotlib

from Metrics import ownPhaseMetric, ownPhaseMetricCorrected, ownPhaseMetric_numpy
from Metrics import ownPhaseMetricCorrected_numpy, ssimMetric, statDistributionNormalized
from Metrics import MSEstatDistribution, MeanPowerSpectrumComparison, MSEPowerSpectrumComparison
from Metrics import Correlation, PowerSpectrumComparison
from ApplyOptimumFilter import ApplyOptimumFilter as OF
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib
#%%

#root=r'D:\DLOCT\Cgansub_260123'
#model_folder = '\cGAN_subsampling'

root=r'D:\DLOCT\Cgansub_260123'
model_folder = '\\Nueva carpeta\cGAN_subsampling'

path = root+model_folder
path2 = r'D:\cGAN_subsampling'
d_loss1 = np.load(path+'\d_loss1.npy')
d_loss2 = np.load(path+'\d_loss2.npy')
g_loss = np.load(path+'\g_loss.npy')
n_epochs = np.load(path+'\\n_epochs.npy')
#%% compare 

# fig,axs = plt.subplots(2,1)
# axs[0].plot(d_loss1)
# axs[0].plot(d_loss2)
# axs[1].plot(g_loss)
# %matplotlib

#%%
customObjects = {'ownPhaseMetric': ownPhaseMetric,
                 'ownPhaseMetricCorrected': ownPhaseMetricCorrected}
model = tf.keras.models.load_model(path+'\\Models\\model_016384.h5',custom_objects=customObjects)
model.summary()
import plotly.io as pio
pio.templates.default = 'presentation'

# figsize with conversion factor from mm to inches
# (338.67, 190.5) is slide size
mm2pixels = 3.7795275591
dpi = 300


#%% """ Defining parameters """

# List of tomograms to use in the DataSet,
# do not include de _real or _imag, nor the extension

# Porcine cornea
# rootFolder = 'D:/DLOCT/TomogramsDataAcquisition/'

# fnameTom = [
#     'ExperimentalTomogram/ExperimentalROI_corrected5',
#     ]
# tomStructs = [0]

# Porcine cornea
rootFolder = 'D:/DLOCT/TomogramsDataAcquisition/'

# fnameTom = [
#     '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomJones_z=(586)_x=(512)_y=(512)',
#     ]
# tomStructs = [0]

fnameTom = [
    'ExperimentalTomogram/ExperimentalROI_corrected5',
    ]
tomStructs = [0]


# Shape of each tomogram, as tuples (Z, X, Y)
#tomShape = [(350,384,384)]
tomShape = [(350,384,384)]
# tomShape = [(586,512,512)] # s.eye_swine
n = 128
testSize = 0.25
slidingXSize = n
slidingYSize = n
strideX = slidingXSize
strideY = slidingYSize

# savedModelPath = r'G:\Data\DLOCT\outputs\ImportantModels\BestAsymmetricAutoencoder'
# savedModelPath = r'G:\Data\DLOCT\outputs\ImportantModels\BestSymmetricAutoencoder'
# savedModelPath = r'G:\Data\DLOCT\outputs\ImportantModels\BestUNet'
# savedModelPath = r'G:\Data\DLOCT\outputs\ImportantModels\UNetplus_DR20'
#savedModelPath = r'D:\DLOCT\ImportantModels\BestUNet'
# savedModelPath = r'C:\Users\diego\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\OCT_Advanced_Postprocessing_copia\Analysis\DLOCT\Saved_model_cGANSub\g_model_short'
savedModelPath = r'D:\DLOCT\Cgansub_260123\Nueva carpeta\cGAN_subsampling\Models'

logModel = True # true if the model is trained with logdata

# savefolder = r'G:\Data\DLOCT\outputs\ModelEvaluationTomogram\AsymmetricAutoencoder'
# savefolder = r'G:\Data\DLOCT\outputs\ModelEvaluationTomogram\SymmetricAutoencoder'
# savefolder = r'G:\Data\DLOCT\outputs\ModelEvaluationTomogram\BestUNet'
# savefolder = r'G:\Data\DLOCT\outputs\ModelEvaluationTomogram\UNetplus_DR20'
savefolder = r'D:\DLOCT\Cgansub_260123\Nueva carpeta\cGAN_subsampling\results2'
saveImages = True
#%%

i = 0
zs = 128
name = 'Experimental'
print('\n\n\n ---- TOMOGRAM: ', str(i), ' ----\n\n\n')

fnameTomdata = dict(
fnameTom=[fnameTom[i]],
tomStructs=[tomStructs[i]],
tomShape=[tomShape[i]])

#% """ Loading data """
    
slices, _, tomData = LoadData(rootFolder, slidingXSize, slidingYSize, strideX, strideY,
                                fnameTomdata=fnameTomdata)

tomData = tomData[0]
logslices, slicesMax, slicesMin = logScaleSlices(slices)
logslicesUnder = downSampleSlices(logslices)
#% Processing undersampled fields with model

    
logslicesOver = np.array(model.predict(logslicesUnder, batch_size=8), dtype='float64')
slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
#%
slicesUnder=downSampleSlices(slices)
#% Quantitative metrics log scale
#%%
print('\n\n\t\t\t --- Quantitative metrics log scale ---\n')

ssims = np.mean(ssimMetric(logslices, logslicesOver))
phasemetric = np.mean(ownPhaseMetric_numpy(logslices, logslicesOver))
phasemetricCorrected = np.mean(ownPhaseMetricCorrected_numpy(logslices, logslicesOver))
mse = np.mean((logslices - logslicesOver)**2)

print('\t\t -- For test data')
print('MSE: ', mse)
print('Phase STD [0, 1]: ', phasemetric)
print('Phase STD [-1, 1]: ', phasemetricCorrected)
print('SSIM: ', ssims)

#% Quantitative metrics normal scale, includes MSE of distributions, and MSE
# of power spectrum

print('\n\n\t\t\t --- Quantitative metrics normal scale ---\n')

# ssims = np.mean(ssimMetric(Y_test, X_test))
# phasemetric = np.mean(ownPhaseMetric_numpy(Y_test, X_test))
# mse = np.mean((Y_test - X_test)**2)
# MSEdistribution = MSEstatDistribution(Y_test, X_test)

# print('\t\t -- For subsampled data')
# print('MSE: ', mse)
# print('Phase STD: ', phasemetric)
# print('SSIM: ', ssims)
# print('MSE statistical distribution: ', MSEdistribution)

ssims = np.mean(ssimMetric(slices, slicesOver))
phasemetric = np.mean(ownPhaseMetric_numpy((slices + 1) / 2,
                                            (slicesOver + 1) / 2 ) )
phasemetricCorrected = np.mean(ownPhaseMetric_numpy(slices, slicesOver))
mse = np.mean((slices - slicesOver)**2)
# MSEdistribution = np.mean(MSEstatDistribution(slices, slicesOver))
# MSEPS = np.mean(MSEPowerSpectrumComparison(slices, slicesOver, meandim=1))

print('\t\t -- For test data')
print('MSE: ', mse)
print('Phase STD [0, 1]: ', phasemetric)
print('Phase STD [-1, 1]: ', phasemetricCorrected)
print('SSIM: ', ssims)
# print('MSE statistical distribution: ', MSEdistribution)
# print('MSE power spectrum: ', MSEPS)


# Stitching
#%%
tomShapex = tomShape[i][1]
tomShapey = tomShape[i][2]
tomShapez = tomShape[i][0]
# zinit=71
# zfinal= zinit + int(np.round(tomShape[0][0]/4))
# tomData=tomData[zinit:zfinal,:,:,:]
# tomShapex = tomData.shape[1]
# tomShapey = tomData.shape[2]
# tomShapez = tomData.shape[0]
sliceid = 0
tomDataOver = np.zeros((tomShapez, tomShapex, tomShapey, 2))
for z in range(tomShapez):
            slidingYPos = 0
            # print(' z dimension :', z)
            while slidingYPos + slidingYSize <= tomShapey:
                slidingXPos = 0
                # print('\t sliding pos y :', slidingYPos)
                while slidingXPos + slidingXSize <= tomShapex:
                    # print('\t\t sliding pos x :', slidingXPos)
                    tomSliceOver = slicesOver[sliceid]
                    tomDataOver[z, slidingXPos: slidingXPos + slidingXSize,
                                        slidingYPos:slidingYPos + slidingYSize, :] = tomSliceOver
                    slidingXPos = slidingXPos + strideX
                    sliceid = sliceid + 1
                slidingYPos = slidingYPos + strideY
#%
tomDataUnder = downSampleSlices(tomData)

#%Plottings and qualitative metrics are validated in whole enface  
#%%
z = 64
print('PLane z = {}'.format(z))

# ownPlot(logY_test[sliceid], givenTitle='Original Field')
# # ownPlot(slicesUnder[100], givenTitle='Downsampled Field')
# ownPlot(logslicesOver_test[sliceid], givenTitle='Upsampled Field')
# ownPlot(logY_test[sliceid] - logslicesOver_test[sliceid], 'Substraction')

if saveImages:
    savename = join(savefolder, 'tom_' + name + 'z_' + str(z))
    savename_dist = savename + '_statDist' + '.svg'
else:
    savename = None
    savename_dist = None

ComparisonPlotField_matplotlib(tomData[z, :, :], tomDataOver[z, :, :],
                                savename=savename, vminInt=53, vmaxInt=75
                                )
tmp1 = statDistributionNormalized(tomData[z, :, :, 0], tomDataOver[z, :, :, 0],
                                    savename=savename_dist, normalize=False,
                                    density=True, fitting=False)

tmp2 = PowerSpectrumComparison(tomData[z, :, :, :], tomDataOver[z, :, :, :],
                                tomDataUnder[z,:,:,:],
                                savename=savename, meandim=1, normalize=True)


Correlation(tomDataOver[z, :, :, :],savename=savename)
#% 
tmp3, MeanPowerSpectrumOver = MeanPowerSpectrumComparison(tomData, tomDataOver,
                                    savename=join(savefolder, 'tom_' + name),
                                    normalize=True, meandim=0)
print('ok')
#%%

z = 64
plot = 10*np.log10(abs(tomDataOver[z,:,:,0] + 1j*tomDataOver[z,:,:,1])**2)
plt.imshow((plot),cmap='gray',vmin=30, vmax=100)
# %matplotlib
#%%
plot = abs(tomData[z,:,:,0] + 1j*tomData[z,:,:,1])
plotemp= MeanPowerSpectrumOver
plotemp[192,192]=0.01
#plt.imshow((plotemp),cmap='viridis')
import plotly.express as px
cmap='viridis'
savename=join(savefolder, 'tom_' + name)
fig = px.imshow(plotemp,
        color_continuous_scale=cmap,
        title='Reconstructed Tomogram Mean Power Spectrum',
        )
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(
    margin=dict(l=10, r=10, t=40, b=20),
    coloraxis_colorbar_x=0.83,
    font_family="Times New Roman",
    font_size=16,)
fig.show(renderer = 'svg+notebook')
fig.write_image(savename + '_Mean_Power_Spectrum_correct.svg')

#%%

vmin = 50
vmax=120
tomDatap,tomIntDonw = OF(tomData, tomDataOver,z,vmin,vmax)

#%%

fig = plt.figure(30)
ax = fig.add_subplot(111)
dum = ax.imshow(10 * np.log10(tomIntDonw[z, :, :]), cmap='gray', vmin=vmin, vmax=vmax)
#ax.set_aspect(2)
ax.set_title('subsampled')
fig.colorbar(dum)