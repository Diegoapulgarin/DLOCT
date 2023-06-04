#%%
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import windows
from CreateLayeredSample import create_Layered_Sample
import matplotlib.pyplot as plt
# from ForwardModel_PointScatterers_FreqLowNA_3D import LowNA_3D
from ForwardModel_PointScatterers_HighNA import HighNA
import time
#%%

def coerce(matrix, minimum=0, maximum=1):
    """
    Coerce matrix into range defined by minimum and maximum. By default
    minimum = 0, maximum = 1.

    Parameters
    ----------
    matrix : ndarray
        Input array.
    minimum : scalar, optional
        Lower limit for coercion.
    maximum : scalar, optional
        Upper limit for coercion.

    Returns
    -------
    matrix_coerced : ndarray
        Coerced array.
    changed : bool
        True if any value had to be coerced.
    """

    # Create an array to track NaN values
    matrix_nan_idx = np.isnan(matrix)

    # Coerce matrix
    matrix_coerced = np.maximum(matrix, minimum)
    matrix_coerced = np.minimum(matrix_coerced, maximum)
  
    # Put NaNs back in
    matrix_coerced[matrix_nan_idx] = np.nan
  
    # Calculate whether any values were changed
    changed = np.any(matrix != matrix_coerced)

    return matrix_coerced, changed

def LowNA_3D(amp, z, x, y, kVect, k, xi_x, xi_y, alpha, zFP, zRef, maxBatchSize=None):
    # Beam waist diameter
    beamWaistDiam = 2 * alpha / k
    # Raylight range
    zR = 2 * alpha ** 2 / k
  
    # Remove all points beyond n times the beam position; their contribution is not worth the calculation
    nullPoints = np.sqrt(x**2 + y**2) > 20 * beamWaistDiam * np.sqrt(1 + (z / zR) ** 2)
    amp = amp[~nullPoints]
    amp = np.reshape(amp,(1,1,np.shape(amp)[0]))
    z = z[~nullPoints]
    z = np.reshape(z,(1,1,np.shape(z)[0]))
    x = x[~nullPoints]
    x = np.reshape(x,(1,1,np.shape(x)[0]))
    y = y[~nullPoints]
    y = np.reshape(y,(1,1,np.shape(y)[0]))
  
    # Number of points
    nPoints = z.shape[2]
    
    # If not input batchSize calculate contribution from all points at once
    if maxBatchSize is None:
        maxBatchSize = nPoints
    
    # Batch size
    batchSize = min(maxBatchSize, nPoints)
    
    # Number of batches of points
    nBatches = np.ceil(nPoints / batchSize)
    
    # Initialize output
    fringes = np.zeros((kVect.shape[0], 1), dtype=kVect.dtype)
    # kVect = kVect[np.newaxis, np.newaxis, np.newaxis, :]
    
    # Iterate batches
    for j in range(int(nBatches)):
        # Calculate the contribution from this batch of points
        thisBatch = np.minimum((np.array(range(batchSize)) + j * batchSize), nPoints)
        print(j)
        print(thisBatch)
        print(np.shape(thisBatch))
        # In this case we use 2*kVect - xi_x^2/(4*k) where kVect is a vector BUT k is an scalar, yielding the low-NA model
        a = 1 / (8 * np.pi ** 2)/((alpha / k) ** 2 + (1j * (z[:,:,thisBatch-1] - zFP) / k))
        b = np.exp(2j * (z[:,:,thisBatch-1] - zRef) * kVect)
        b = np.transpose(b,(1,0,2))
        c = np.sum(np.exp(-1j * ( xi_x * x[:,:,thisBatch-1] ))*
                   np.exp(-1j * (z[:,:,thisBatch-1] - zFP) * xi_x ** 2 / k / 4) *
                   np.exp(- (xi_x * alpha / k / 2) ** 2),axis=0, keepdims=True)
        da = np.exp(-1j * ( xi_y.T * y[:,:,thisBatch-1] ))
        db = np.exp(-1j * (z[:,:,thisBatch-1] - zFP).T* xi_y ** 2 / k / 4)
        dc = np.exp(- (xi_y * alpha / k / 2) ** 2)
        da = np.transpose(da,(1,2,0))
        db = np.transpose(db,(1,0,2))
        d = np.sum(da*db*dc,axis=2, keepdims=True)
        d = np.transpose(d,(0,2,1))
        thisFringes = a*b*c*d
        # d = np.sum(np.exp(-1j * ( xi_y.T * y[:,:,thisBatch] )) *
        #            np.exp(-1j * (z[:,:,thisBatch] - zFP).T* xi_y ** 2 / k / 4) *
        #            np.exp(- (xi_y * alpha / k / 2) ** 2),axis=2, keepdims=True)
        # thisFringes = 1 / (8 * np.pi ** 2) / \
        #     ((alpha / k) ** 2 + (1j * (z[:,:,thisBatch] - zFP) / k)) * \
        #         np.exp(2j * (z[:,:,thisBatch] - zRef) * kVect) * \
        #             np.sum(np.exp(-1j * ( xi_x * x[:,:,thisBatch] )) * \
        #                    np.exp(-1j * (z[:,:,thisBatch] - zFP) * xi_x ** 2 / k / 4) * \
        #                     np.exp(- (xi_x * alpha / k / 2) ** 2),axis=1, keepdims=True) * \
        #                         np.sum(np.exp(-1j * ( xi_y * y[:,:,thisBatch] )) * \
        #                                np.exp(-1j * (z[:,:,thisBatch] - zFP)* xi_y ** 2 / k / 4) * \
        #                                 np.exp(- (xi_y * alpha / k / 2) ** 2),axis=2, keepdims=True)
        # sum the contribution of all scatteres, considering its individual amplitudes
        fringes = fringes + np.sum(amp[:,:,thisBatch-1] * thisFringes, axis=2)
        fringes = fringes[:,0]
        print('ok')
    return fringes


#%%
varType = 'float32'
# Parámetros de la simulación
# Número de puntos del tomograma
nZ = 32  # axial, número de píxeles por línea A, teniendo en cuenta el relleno con ceros
nX = 32 + 32  # eje de exploración rápida, número de líneas A por exploración B
nY = 1  # eje de exploración lenta, número de exploraciones B por tomograma
nK = 128  # Número de muestras, <= nZ, la diferencia es el relleno con ceros
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
freqXVect = np.reshape(freqXVect,(1,1,len(freqXVect))).T
nFreqY = nY * freqBWFac
freqYVect = np.zeros((1,1,2))
freqYVect[0,0,:] = np.float32(np.linspace(-0.5, 0.5 - 1 / nFreqY, nFreqY)) / (latSampling / freqBWFac) * 2 * np.pi

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

# Crea un array de ceros con las dimensiones especificadas
objSuscep = np.zeros((nZ, nX))

# Asegúrate de que las posiciones estén dentro de los límites
objPosZ_rounded = np.clip(np.round(objPosZ / axSampling).astype(int), 0, nZ-1)
objPosX_rounded = np.clip(np.round(objPosX / latSampling).astype(int) + nX // 2, 0, nX-1)

#%%
# Suponiendo que varType es 'float' en este contexto
objSuscep = np.zeros((nZ, nX, nY))

# Coerción de los índices
# Para evitar la asignación de NaNs a matrices enteras, 
# se puede mantener todo en flotante y luego convertir a int después de la coerción
coerced_objPosZ, _ = coerce(np.round(objPosZ / axSampling), 1, nZ)
coerced_objPosX, _ = coerce(np.round(objPosX / latSampling) + nX / 2, 1, nX)

# Redondeamos y convertimos a enteros después de la coerción para evitar el error de NaN
coerced_objPosZ = np.round(coerced_objPosZ).astype(int)
coerced_objPosX = np.round(coerced_objPosX).astype(int)

# Creación de índices para un array 3D y obtención de índices únicos
# Asegurémonos de que estamos dando tres dimensiones a np.ravel_multi_index
objSuscepIndx = np.ravel_multi_index((coerced_objPosZ-1, coerced_objPosX-1, np.zeros_like(coerced_objPosZ, dtype=int)), objSuscep.shape)
objSuscepIndx_unique, objAmpIndx = np.unique(objSuscepIndx, return_index=True)
objAmp_flat = objAmp.ravel()
np.put(objSuscep, objSuscepIndx_unique, objAmp_flat[objAmpIndx])
#%%
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
kVect = kVect.reshape((128, 1))  # Changing shape of k_vect to (128, 1, 1)
if modelISAM:
    print('High NA')
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
    print('Low NA')
    for thisYScan in range(nY):
        for thisXScan in range(nX):
            # Current beam position
            thisBeamPosX = xVect[thisXScan]
            thisBeamPosY = yVect[thisYScan]
            # Spectrum at this beam possition is the contribution of the Gaussian
            # beam at the location of the point 
            # fringes1= LowNA_3D(
            #     objAmp, objPosZ, objPosX - thisBeamPosX, objPosY - thisBeamPosY, 
            #     kVect, wavenumber, freqXVect, freqYVect, alpha, focalPlane, 
            #     zRef, maxPointsBatch)
            fringes1[:, thisXScan, thisYScan] = LowNA_3D(
                objAmp, objPosZ, objPosX - thisBeamPosX, objPosY - thisBeamPosY, 
                kVect, wavenumber, freqXVect, freqYVect, alpha, focalPlane, 
                zRef, maxPointsBatch)



print("Execution time:", time.time() - start_time)
# Calculate fringes with proper constants, including source spectrum
sourceSpec1 = np.reshape(sourceSpec, (len(sourceSpec),1,1))
kVect = np.reshape(kVect, (len(kVect),1,1))
fringes = fringes1 * 1j / ((2 * np.pi) ** 2) * 1 * np.sqrt(sourceSpec1) / kVect
#%%
tom1 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(fringes)))
plt.imshow(abs(tom1)**2)


#%%
# # Beam waist diameter
# beamWaistDiam = 2 * alpha / wavenumber
# # Raylight range
# zR = 2 * alpha ** 2 / wavenumber
# x = objPosX - thisBeamPosX
# y = objPosY - thisBeamPosY
# z = objPosZ
# amp = objAmp
# # Remove all points beyond n times the beam position; their contribution is not worth the calculation
# nullPoints = np.sqrt(x**2 + y**2) > 20 * beamWaistDiam * np.sqrt(1 + (z / zR) ** 2)
# amp = amp[~nullPoints]
# amp = np.reshape(amp,(1,1,np.shape(amp)[0]))
# z = z[~nullPoints]
# z = np.reshape(z,(1,1,np.shape(z)[0]))
# x = x[~nullPoints]
# x = np.reshape(x,(1,1,np.shape(x)[0]))
# y = y[~nullPoints]
# y = np.reshape(y,(1,1,np.shape(y)[0]))
# #%%
# da = np.transpose(da,(1,2,0))
# db = np.transpose(db,(1,0,2))
# #%%
# d = np.sum(da*db*dc,axis=2, keepdims=True)
# d = np.transpose(d,(0,2,1))

# #%%
# nfringes = a*b*c*d

# test = fringes1[:,0]