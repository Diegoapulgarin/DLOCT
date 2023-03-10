# -*- coding: utf-8 -*-
"""
Cargar y revisar los datos de tomogramas experimentales
"""
from os.path import join

# Custom modules
import sys
sys.path.append(r'C:\Users\labfisica\Documents\Intercambio_Informacion_EAFIT\Analysis\DLOCT\TrainingModels')

from Utils import LoadData, ComparisonPlotField_matplotlib 
from Metrics import Correlation, statDistributionNormalized

import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
#%% """ Defining parameters """

savefolder = r'C:\Users\diego\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\OCT_Real\ExperimentalTomogram'
# Lab's computer
rootFolder = 'C:/Users/diego/OneDrive - Universidad EAFIT/Eafit/Trabajo de grado/OCT_Real/'

# Own laptop
# rootFolder = 'C:/Users/josehernan/Desktop/outputs/TomogramsDataAcquisition/'

fnameTom = [
    'ExperimentalTomogram/ExperimentalROI5',
    'ExperimentalTomogram/ExperimentalROI_corrected5',
    ]

tomStructs = [0, 1]
tomShape = [(350,384,384), (350,384,384)]

fnameTomdata = dict(
    fnameTom=fnameTom,
    tomStructs=tomStructs,
    tomShape=tomShape)


slidingXSize = 128
slidingYSize = 128
strideX = slidingXSize
strideY = slidingYSize

inputShape = (slidingXSize, slidingYSize, 2)


#%% """ Loading data """

_, _, tomDatas = LoadData(rootFolder, slidingXSize, slidingYSize, strideX, strideY,
                        fnameTomdata=fnameTomdata)

z = 32

tomData0 = tomDatas[0]
tomData1 = tomDatas[1]
tomDatas.append(tomData1[:, 1::2, :, :])
tomData2 = tomDatas[2]

ComparisonPlotField_matplotlib(tomData0[z, 64:128+64, :128, :], tomData1[z, 64:128+64, :128, :],
                               savename=join(savefolder, 'EnfacesComparisonOriginalvsprocessed'))

EnfaceInt = abs(tomData0[z,:,:,0] + 1j*tomData0[z,:,:,1])**2
fig = plt.figure()
plt.imshow(np.log10(EnfaceInt), cmap = 'gist_gray', vmin=None, vmax=None)
plt.colorbar()
plt.title('Enface ' + str(z))
plt.axis('off')
fig.savefig(join(savefolder, 'Enface' + str(z) + '.pdf'))

comments = ['original', 'processed', 'down']
for i, tomData in enumerate(tomDatas):
    comment = comments[i]
    Correlation(tomData[z, :, :, :], savename=join(savefolder, 'Enface'+str(z) + comment))
    
#%% 
