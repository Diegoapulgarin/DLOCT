#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.fft import fft,fftshift, ifft
def fast_reconstruct(array):
    tom = fftshift(fft(fftshift(array,axes=0),axis=0),axes=0)
    return tom
 
def extract_dimensions(file_name):
    parts = file_name.split('_')
    dimensions = []
    for part in parts:
        if 'z=' in part or 'x=' in part or 'y=' in part:
            number = int(part.split('=')[-1])
            dimensions.append(number)
    return tuple(dimensions)

def read_tomogram(file_path, dimensions):
    depth, height, width = dimensions
    with open(file_path, 'rb') as file:
        tomogram = np.fromfile(file, dtype='single')
        tomogram = tomogram.reshape((depth, height, width),order='F')
    return tomogram
def plot(bscan):
     plt.imshow(10*np.log10(abs(bscan)**2),cmap='gray')

def identify_max_component(spectrum):
    """
    Identifica el componente máximo en el espectro dado.
    
    :param spectrum: El espectro (ci1 o ci2) como un array de NumPy.
    :return: índice(s) y valor del componente máximo.
    """
    max_value = np.max(spectrum)
    max_index = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    return max_index, max_value

def update_outputs_and_subtract_artifacts(d1, d2, ci1, ci2, max_component_1_index, max_component_2_index):
    """
    Actualiza los espectros y las salidas basándose en la eliminación de artefactos.
    
    :param d1: Salida acumulativa para ci1.
    :param d2: Salida acumulativa para ci2.
    :param ci1: Espectro compensado por dispersión.
    :param ci2: Espectro no compensado por dispersión.
    :param max_component_1_index: Índice del componente máximo en ci1.
    :param max_component_2_index: Índice del componente máximo en ci2.
    :return: Los espectros y las salidas actualizadas.
    """
    # Ejemplo simplificado: sustraer el componente máximo de cada espectro
    ci1[max_component_1_index] -= ci1[max_component_1_index]  # Ajustar según necesidades
    ci2[max_component_2_index] -= ci2[max_component_2_index]  # Ajustar según necesidades
    
    # Aquí se deberían implementar los ajustes reales basados en los detalles específicos del proceso
    
    return d1, d2, ci1, ci2

#%%
path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Data Boston\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'
file = '[p.Calibration][s.Mirror][02-10-2023_15-17-52].dispersion'
dispersion = np.fromfile(os.path.join(path,file))
plt.plot(dispersion)
path = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscan'
artifact_files = os.listdir(path)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(path, real_file)
        imag_file_path = os.path.join(path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        del tomImag, tomReal
# fringes = fftshift(ifft(fftshift(tom,axes=0),axis=0),axes=0)
#%%

phase_correction = np.exp(-1j * dispersion)[:, np.newaxis, np.newaxis]
corrected_signal = tom * phase_correction
corrected_volume =  fftshift(ifft(fftshift(corrected_signal,axes=0),axis=0),axes=0)
# Convertir el volumen a magnitud para la detección de picos
volume_magnitude = np.abs(corrected_volume)
# Definir un umbral para la detección de picos
threshold = np.mean(volume_magnitude) + 2 * np.std(volume_magnitude)
# Detectar picos
is_peak = volume_magnitude > threshold
# Eliminar picos: Establecer los valores detectados como picos a un valor base, por ejemplo, cero o el valor mínimo del volumen
corrected_volume[is_peak] = np.min(volume_magnitude)
#%%
volume = fftshift(ifft(fftshift(tom,axes=0),axis=0),axes=0)

# Parámetros iniciales
M = 10  # Número máximo de iteraciones
T = 0.01  # Umbral para la detección de artefactos significativos

# Inicialización de variables
i = 0  # Índice de iteración
d1 = np.zeros_like(corrected_volume)  # Salida para componentes de señal
d2 = np.zeros_like(volume)  # Salida para componentes de autocorrelación (si es aplicable)

while i < M:
    # Identificar los componentes de señal con mayor contribución
    max_component_1_index,max_component_1 = identify_max_component(corrected_volume)
    max_component_2_index,max_component_2 = identify_max_component(volume)  # Si aplica

    # Comparar con el umbral y decidir si proceder
    if max_component_1 < T and max_component_2 < T:  # Ajustar condición según corresponda
        break  # Salir del bucle si no hay componentes significativos

    # Actualizar las salidas y restar los artefactos correspondientes
    d1, d2, corrected_volume, volume = update_outputs_and_subtract_artifacts(d1, d2, corrected_volume, volume, max_component_1, max_component_2)

    i += 1  # Incrementar el índice de iteración

# Opcional: añadir el espectro restante a la salida
# d1 += ci1
# d2 += ci2  # Si aplica



