import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import windows
import CreateLayeredSample
import ForwardModel_PointScatterers_FreqLowNA_3D
import ForwardModel_PointScatterers_HighNA

# Parámetros de la simulación
nZ = 64
nX = 64
nY = 1
nK = 256
xNyquistOversampling = 1
nXOversampling = nX

useGPU = False

# Parámetros espectrales
wavelength = 1.310e-6
wavelengthWidthSource = 1*120e-9
axialRes = 2 * np.log(2) / np.pi * wavelength ** 2 / wavelengthWidthSource

# Parámetros confocales
numAper = 0.05

# Nivel de ruido en dB
noiseFloorDb = 20

if useGPU:
  # Aquí se podría utilizar bibliotecas como cupy o tensorflow para cálculos en GPU
  pass
else:
  ToSingle = lambda x: np.float32(x)

# Rango espectral del número de onda
wavenumberRange = 2 * np.pi / (wavelength + np.array([wavelengthWidthSource, -wavelengthWidthSource]) / 2)
zeroPadding = (nZ - nK) // 2
kVect = np.linspace(wavenumberRange[0], wavenumberRange[1], nK)
wavenumber = (wavenumberRange[0] + wavenumberRange[1]) / 2

# Resto del código...
# Parámetros del muestreo
xVect = np.linspace(-nX/2, nX/2-1, nXOversampling)
yVect = 0
zVect = np.linspace(0, nZ-1, nZ)
zz, xx, yy = np.meshgrid(zVect, xVect, yVect)

# Parámetros de los scatterers
nScatterers = 2
scattererPositions = np.array([[0, 0, 1.5*axialRes], [0, 0, 2.5*axialRes]])
scattererAmplitudes = np.array([1, -1])

# Coeficiente de backscattering
muBackscat = 1

# Modelado de campo de backscatter (HighNA)
EBackscat = ForwardModel_PointScatterers_HighNA(scattererPositions, scattererAmplitudes, 
                                                xx, yy, zz, wavenumber, numAper, nZ, nX, nY)
EBackscat = muBackscat * EBackscat

# Modelado de campo de backscatter (LowNA)
kxVect = np.fft.fftfreq(nX, d=xVect[1] - xVect[0]) * 2 * np.pi
kyVect = np.fft.fftfreq(nY, d=yVect[1] - yVect[0]) * 2 * np.pi
EBackscatLowNA = ForwardModel_PointScatterers_FreqLowNA_3D(kxVect, kyVect, kVect, 
                                                            scattererPositions, scattererAmplitudes, nZ, nX, nY)
EBackscatLowNA = muBackscat * EBackscatLowNA

# Backscattered field with speckle
EBackscatSpeckle = np.zeros((nZ, nXOversampling, nY), dtype=np.complex64)
EBackscatSpeckle[zeroPadding:zeroPadding+nK, :, :] = EBackscat

# Field from layered sample
nLayers = 5
layerPositions = np.linspace(0, nZ-1, nLayers)
layerRefrIndices = 1.4 * np.ones(nLayers)
layerThickness = np.mean(np.diff(layerPositions))
layerRefrIndices = CreateLayeredSample(nZ, layerPositions, layerRefrIndices, layerThickness)

# FFT along the axial direction
FFT_EBackscat = np.fft.fft(EBackscatSpeckle, axis=0)
FFT_EBackscat = fftshift(FFT_EBackscat, axes=0)

# Convert the signal to dB scale
FFT_EBackscat_Db = 20 * np.log10(np.abs(FFT_EBackscat))

# Plot the result
plt.imshow(FFT_EBackscat_Db, extent=[xVect[0], xVect[-1], zVect[0], zVect[-1]], aspect='auto', cmap='jet')
plt.title('FFT of Backscattered Field (in dB)')
plt.xlabel('Lateral Position')
plt.ylabel('Axial Position')
plt.colorbar(label='Amplitude (dB)')
plt.show()
