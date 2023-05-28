#%%
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import windows
from CreateLayeredSample import create_Layered_Sample
import matplotlib.pyplot as plt
from ForwardModel_PointScatterers_FreqLowNA_3D import LowNA_3D
from ForwardModel_PointScatterers_HighNA import HighNA
import time
#%%
varType = 'float32'
# Parámetros de la simulación
# Número de puntos del tomograma
nZ = 512  # axial, número de píxeles por línea A, teniendo en cuenta el relleno con ceros
nX = 256 + 32  # eje de exploración rápida, número de líneas A por exploración B
nY = 1  # eje de exploración lenta, número de exploraciones B por tomograma
nK = 400  # Número de muestras, <= nZ, la diferencia es el relleno con ceros
xNyquistOversampling = 1  # Factor de muestreo del galvanómetro. 1 -> Nyquist
nXOversampling = nX  # Número de líneas A para sobremuestreo de la PSF <= nX, la diferencia es el relleno con ceros

useGPU = False

# Parámetros espectrales
wavelength = 1.310e-6  # Longitud de onda central de la fuente
wavelengthWidthSource = 1 * 120e-9  # Anchura espectral completa a la mitad del máximo
axialRes = 2 * np.log(2) / np.pi * wavelength ** 2 / wavelengthWidthSource  # Resolución axial

# Parámetros confocales
numAper = 0.05  # Apertura numérica

# Nivel de piso de ruido en dB
noiseFloorDb = 20

#%%
# Rango espectral del número de onda
wavenumberRange = 2 * np.pi / (wavelength + (np.array([wavelengthWidthSource, -wavelengthWidthSource]) / 2))

# Vector de muestreo del número de onda. Debido a que estamos simulando franjas complejas, necesitamos nZ y no 2 * nZ
zeroPadding = (nZ - nK) / 2
kVect = np.linspace(wavenumberRange[0], wavenumberRange[1], nK)
wavenumber = np.float32((wavenumberRange[0] + wavenumberRange[1]) / 2)

# Ancho espectral del número de onda de la fuente
wavenumberWidthSource = 2 * np.pi / (wavelength - (wavelengthWidthSource / 2)) - 2 * np.pi / (wavelength + (wavelengthWidthSource / 2))

# Espectro lineal en el número de onda de la fuente
wavenumberFWHMSource = wavenumberWidthSource / (2 * np.sqrt(2 * np.log(2)))
sourceSpec = np.exp(-(kVect - wavenumber) ** 2 / 2 / wavenumberFWHMSource ** 2)

# Tamaño físico del eje axial
zSize = np.pi * nK / np.diff(wavenumberRange)
axSampling = zSize / nZ  # Muestreo axial

# Parámetros confocales
alpha = np.pi / numAper  # Constante confocal
beamWaistDiam = 2 * alpha / wavenumber  # Diámetro de la cintura del haz
latSampling = beamWaistDiam / 2 / xNyquistOversampling  # Muestreo lateral
confocalParm = np.pi * (beamWaistDiam / 2) ** 2 / wavelength  # Parámetro confocal (para información.)

# Retardo de cero camino. Cambiando esto cambia el plano focal en el modelo HighNA
zRef = zSize / 2
# Distancia desde el plano superior al plano focal. Esto es independiente de
# zRef solo para los modelos LowNA y no para el modelo HighNA
focalPlane = zSize / 4

xSize = latSampling * nX  # Tamaño físico del eje de escaneo rápido
ySize = latSampling * nY  # Tamaño físico del eje de escaneo lento

# Coordenadas
# Coordenada cartesiana
zVect = np.float32(np.linspace(0, zSize - axSampling, nZ))
xVect = np.float32(np.linspace(-xSize / 2, xSize / 2 - latSampling, nX))
yVect = np.linspace(-ySize / 2, ySize / 2 - latSampling, nY); np.float32(0)

# Coordenadas de frecuencia
freqBWFac = 2  # Aumentar el ancho de banda de frecuencia para evitar artefactos en la FT numérica
nFreqX = nX * freqBWFac
freqXVect = np.float32(np.linspace(-0.5, 0.5 - 1 / nFreqX, nFreqX)) / (latSampling / freqBWFac) * 2 * np.pi

nFreqY = nY * freqBWFac
freqYVect = np.float32(np.linspace(-0.5, 0.5 - 1 / nFreqY, nFreqY)) / (latSampling / freqBWFac) * 2 * np.pi
#%%

# Parámetros para crear el objeto con dispersores puntuales
nPointSource = nY * 50000  # asegúrate de que nPointSource/nY es un número entero
maxPointsBatch = round(nPointSource/16)

# Rango donde aparecen los dispersores puntuales
layersPointSources = 0  # [4, 6]
zStart = 32
zOffset = 0
objZRange = nZ - 32
objXRange = nX - 32
objYRange = nY - 0
objRange = [objZRange, objXRange]

layeredObj = False
#%%

if layeredObj:
    # Porcentaje de longitud de cada capa
    layerPrcts = [5, 20, 5, 10, 5, 5, 2, 5, 43]
    # Intensidad de señal de cada capa
    layerBackScat = np.array([5, 1, 0.5, 1, 0.5, 10, 5, 10, 5]) * 1e-4
    layerScat = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) / 2  # [1, 5, 2, 1]
    # Vector de muestreo
    sampling = [axSampling, latSampling]
    # Creado muestra en capas
    objPos, objAmp, layerBounds = create_Layered_Sample(layerPrcts, layerBackScat, layerScat, layersPointSources, nPointSource//nY, objRange, zStart, sampling, varType, maxPointsBatch)
    objPosZ = objPos[0, :, :] - zOffset * axSampling
    objPosX = objPos[1, :, :]
else:
    objPosZ = (objZRange * np.random.rand(1, 1, nPointSource) - objZRange / 2) * axSampling
    objPosX = (objXRange * np.random.rand(1, 1, nPointSource) - objXRange / 2) * latSampling
    # Amplitud de los dispersores
    objAmp = np.ones((1, 1, nPointSource))  # Todos unos por defecto

#%%

# Adding next B-scans as copies of the initial one
# del objPosY
objPosY = np.zeros((1, 1, nPointSource))
objPosY[0, 0, :] = np.repeat(yVect, nPointSource // nY)

for i in range(1, nY):
    objAmp = np.concatenate((objAmp, objAmp[:, :, :nPointSource//nY]), axis=2)
    objPosX = np.concatenate((objPosX, objPosX[:, :, :nPointSource//nY]), axis=2)
    objPosZ = np.concatenate((objPosZ, objPosZ[:, :, :nPointSource//nY]), axis=2)
    objSuscep = np.concatenate((objSuscep, np.expand_dims(objSuscep[:, :, 0], axis=2)), axis=2)
#%%
# Forward Model
modelISAM = False
start_time = time.time()

# Prepare an empty array for fringes
fringes1 = np.zeros((nK, nX, nY), dtype=varType)

if modelISAM:
    for thisScan in range(nX):
        # Current beam position
        thisBeamPosX = xVect[thisScan]
        # Spectrum at this beam possition is the contribution of the Gaussian
        # beam at the location of the point sources
        fringes1[:, thisScan] = HighNA(
            objAmp, objPosZ, objPosX - thisBeamPosX, 
            kVect, freqXVect, alpha, focalPlane, zRef)
else:
    # Low NA Model
    for thisYScan in range(nY):
        for thisXScan in range(nX):
            # Current beam position
            thisBeamPosX = xVect[thisXScan]
            thisBeamPosY = yVect[thisYScan]
            # Spectrum at this beam possition is the contribution of the Gaussian
            # beam at the location of the point sources
            fringes1[:, thisXScan, thisYScan] = LowNA_3D(
                objAmp, objPosZ, objPosX - thisBeamPosX, objPosY - thisBeamPosY, 
                kVect, wavenumber, freqXVect, freqYVect, alpha, focalPlane, 
                zRef, maxPointsBatch)

# Calculate fringes with proper constants, including source spectrum
fringes1 = fringes1 * 1j / ((2 * np.pi) ** 2) * 1 * np.sqrt(sourceSpec) / kVect

print("Execution time:", time.time() - start_time)