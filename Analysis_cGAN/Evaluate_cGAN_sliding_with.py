# %%
from os.path import join

# Custom imports
import sys
sys.path.append(r'C:\Users\diego\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\OCT_Advanced_Postprocessing_copia\Analysis\DLOCT\cGAN_subsampling')
from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices
from Utils import ComparisonPlotField_matplotlib
from Metrics import ownPhaseMetric, ownPhaseMetricCorrected, ownPhaseMetric_numpy
from Metrics import ownPhaseMetricCorrected_numpy, ssimMetric, statDistributionNormalized
from Metrics import MSEstatDistribution, MeanPowerSpectrumComparison, MSEPowerSpectrumComparison
from Metrics import Correlation, PowerSpectrumComparison
from ApplyOptimumFilter import ApplyOptimumFilter as OF
from Deep_Utils import sliding_window, inv_sliding_window
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
root=r'D:\DLOCT\Cgansub_260123'
model_folder = '\\Nueva carpeta\cGAN_subsampling'
path = root+model_folder
savefolder = r'D:\DLOCT\Cgansub_260123\Nueva carpeta\cGAN_subsampling\results'
d_loss1 = np.load(path+'\d_loss1.npy')
d_loss2 = np.load(path+'\d_loss2.npy')
g_loss = np.load(path+'\g_loss.npy')
n_epochs = np.load(path+'\\n_epochs.npy')
fig,axs = plt.subplots(2,1)
axs[0].plot(d_loss1)
axs[0].plot(d_loss2)
axs[1].plot(g_loss)
#%matplotlib

# %%
customObjects = {'ownPhaseMetric': ownPhaseMetric,
                 'ownPhaseMetricCorrected': ownPhaseMetricCorrected}
model = tf.keras.models.load_model(path+'\\Models\\model_016384.h5',custom_objects=customObjects)
#model.summary()
import plotly.io as pio
pio.templates.default = 'presentation'

# figsize with conversion factor from mm to inches
# (338.67, 190.5) is slide size
mm2pixels = 3.7795275591
dpi = 300

# %%
""" Load tomograms"""
rootFolder = 'D:/DLOCT/TomogramsDataAcquisition/' # porcine cornea
fnameTom = 'ExperimentalTomogram/ExperimentalROI_corrected5' # porcine cornea
# rootFolder = 'D:/DLOCT/TDG/OCT_Real/nueva data/' # s.eye_swine
# fnameTom = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomJones_z=(586)_x=(512)_y=(512)'# s.eye_swine
# Shape of each tomogram, as tuples (Z, X, Y)
tomShape = [(350,384,384)]# porcine cornea
# tomShape = [(586,512,512)] # s.eye_swine
testSize = 0.25
logModel = True # true if the model is trained with logdata
saveImages = True

# %%

name = 'Experimental'
fname = rootFolder + fnameTom
# Names of all real and imag .bin files
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]

# %%
tomReal = np.fromfile(fnameTomReal[0]) # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using
# Fortran style to import according to MATLAB

tomImag = np.fromfile(fnameTomImag[0])
tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using
# Fortran style to import according to MATLAB

tomData = np.stack((tomReal, tomImag), axis=3)
# tomData = tomData/np.max(abs(tomData)) # normalize tomogram


# %%
z = 128
plt.imshow(10*np.log10(abs(tomData[z,:,:,0]+1j*tomData[z,:,:,1])**2))

# %%
n = 128
n1 = 128
window_size = (n,n)
stride = (n1,n1)
slices = []
# zinit=71
# zfinal= zinit + int(np.round(tomShape[0][0]/4))
# tomData=tomData[zinit:zfinal,:,:,:]
for b in range(len(tomData)):
    # i = b + zinit
    i = b 
    bslicei = sliding_window(tomImag[i,:,:],window_size,stride)
    bslicer = sliding_window(tomReal[i,:,:],window_size,stride)
    bslice = np.stack((bslicer,bslicei),axis=3)
    slices.append(bslice)
slices = np.array(slices)
slices = np.reshape(slices,(slices.shape[0]*slices.shape[1],slices.shape[2],slices.shape[3],slices.shape[4]))
del bslicei,bslicer,bslice,tomImag,tomReal

# %%
logslices, slicesMax, slicesMin = logScaleSlices(slices)
logslicesUnder = downSampleSlices(logslices)

# %%
z = 2400
fig,axs = plt.subplots(1,3)
axs[0].imshow(10*np.log10(abs(slices[z,:,:,0]+1j*slices[z,:,:,1])**2))
axs[0].set_title('Normal scale')
axs[0].set_axis_off()
axs[1].imshow((abs(logslices[z,:,:,0]+1j*logslices[z,:,:,1])**2))
axs[1].set_title('log scale')
axs[1].set_axis_off()
axs[2].imshow((abs(logslicesUnder[z,:,:,0]+1j*logslicesUnder[z,:,:,1])**2))
axs[2].set_title('log scale under')
axs[2].set_axis_off()
#fig.suptitle('This is a somewhat long figure title', fontsize=16)

# %%
logslicesOver = np.array(model.predict(logslicesUnder, batch_size=8), dtype='float64')
slicesOver = inverseLogScaleSlices(logslicesOver, slicesMax, slicesMin)
# slicesUnder=downSampleSlices(slices)

# %%
z = 2400
fig,axs = plt.subplots(1,3)
axs[0].imshow(10*np.log10(abs(slicesOver[z,:,:,0]+1j*slicesOver[z,:,:,1])**2))
axs[0].set_title('Normal scale')
axs[0].set_axis_off()
axs[1].imshow((abs(logslicesOver[z,:,:,0]+1j*logslicesOver[z,:,:,1])**2))
axs[1].set_title('log scale')
axs[1].set_axis_off()
axs[2].imshow((abs(slicesUnder[z,:,:,0]+1j*slicesUnder[z,:,:,1])**2))
axs[2].set_title('Normal scale under')
axs[2].set_axis_off()

# %%
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

ssims = np.mean(ssimMetric(slices, slicesOver))
phasemetric = np.mean(ownPhaseMetric_numpy((slices + 1) / 2,
                                            (slicesOver + 1) / 2 ) )
phasemetricCorrected = np.mean(ownPhaseMetric_numpy(slices, slicesOver))
mse = np.mean((slices - slicesOver)**2)
print('\t\t -- For test data')
print('MSE: ', mse)
print('Phase STD [0, 1]: ', phasemetric)
print('Phase STD [-1, 1]: ', phasemetricCorrected)
print('SSIM: ', ssims)



# %%

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
#%%
z = 256
plt.imshow(10*np.log10(abs(tomDataOver[z,:,:,0]+1j*tomDataOver[z,:,:,1]**2)))


# %%
tomDataUnder = downSampleSlices(tomData[0:original_planes,:,:,:])

# %%

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


# %%
z = 64
plot = 10*np.log10(abs(tomDataOver[z,:,:,0] + 1j*tomDataOver[z,:,:,1])**2)
plt.imshow((plot),cmap='gray',vmin=30, vmax=100)
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

# %%
vmin = 50
vmax=120
tomDatap,tomIntDonw = OF(tomData, tomDataOver,z,vmin,vmax)
fig = plt.figure(30)
ax = fig.add_subplot(111)
dum = ax.imshow(10 * np.log10(tomIntDonw[z, :, :]), cmap='gray', vmin=vmin, vmax=vmax)
#ax.set_aspect(2)
ax.set_title('subsampled')
fig.colorbar(dum)

# %%
tomdatapabs = abs(tomDatap)**2
tomdataorabs = abs(tomData[:,:,:,0]+tomData[:,:,:,1])**2
tomdataplog = 10*np.log10(abs(tomDatap)**2)
tomdataslog =  10 * np.log10(abs(tomData[:, :, :, 0] + 1j*tomData[:, :, :, 1])**2)
import tifffile as tif
path = savefolder
tif.imwrite(path+'\\reconstruct2seye.tif', tomdatapabs)
tif.imwrite(path+'\\originalseye.tif', tomdataorabs)
tif.imwrite(path+'\\reconstructlogseye.tif', tomdataplog)
tif.imwrite(path+'\\original_logseye.tif', tomdataslog)


