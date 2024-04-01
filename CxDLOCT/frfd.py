#%%
import os
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm # for progress bars
from frft import frft
from numpy.fft import fft,fftshift, ifft
import matplotlib.pyplot as plt
#%%

def calculate_gradient(signal, alpha, frft, delta=1e-5):
    """
    Calcula el gradiente de la función objetivo usando diferencias finitas.
    """
    frft_plus = frft(signal, alpha + delta)
    frft_minus = frft(signal, alpha - delta)
    gradient = (np.max(np.abs(frft_plus)**2) - np.max(np.abs(frft_minus)**2)) / (2 * delta)
    return gradient

def quasi_newton_method(signal, initial_alpha, frft, max_iterations=10, tol=1e-6, learning_rate=1e-2):
    """
    Método cuasi-Newton para optimizar el valor de alpha.
    """
    alpha = initial_alpha
    for _ in range(max_iterations):
        gradient = calculate_gradient(signal, alpha, frft)
        alpha_next = alpha - learning_rate * gradient
        if np.abs(alpha - alpha_next) < tol:
            break
        alpha = alpha_next
    
    frft_result = frft(signal, alpha)
    power_spectrum = np.abs(frft_result)**2
    peak_position = np.argmax(power_spectrum)
    return alpha, peak_position


def apply_phase_correction(signal, alpha, omega_0):
    """
    Aplica la corrección de fase a la señal utilizando el coeficiente de ajuste derivado de alpha.
    
    :param signal: La señal a corregir.
    :param alpha: El valor de alpha obtenido de la optimización.
    :param omega_0: La frecuencia central de la señal.
    :return: La señal corregida.
    """
    k_n = np.pi * 1/np.tan(alpha)
    corrected_phase = -k_n * (np.arange(len(signal)) - omega_0) ** 2
    return signal * np.exp(1j * corrected_phase)

def apply_dispersion_correction(oct_volume, frft, omega_0, alpha_step=0.5, max_iterations=10, tol=1e-6):
    """
    Aplica corrección de dispersión usando FRFT en un volumen OCT 3D.
    """
    corrected_volume = np.zeros_like(oct_volume, dtype=np.complex)
    
    # Rango de alfas para la búsqueda gruesa
    alphas = np.arange(0, 2 + alpha_step, alpha_step)

    for y in range(oct_volume.shape[2]):
        print(y)
        for x in range(oct_volume.shape[1]):
            aline = oct_volume[:, x, y]
            sa = aline + 1j * hilbert(aline)  # Construcción de la señal analítica
            
            # Búsqueda Gruesa para encontrar alpha_co
            max_peak = 0
            alpha_co = 0  # Corrección: Definir alpha_co antes de la búsqueda
            for alpha in alphas:
                frft_result = frft(sa, alpha)
                power_spectrum = np.abs(frft_result)**2
                max_power = np.max(power_spectrum)
                if max_power > max_peak:
                    max_peak = max_power
                    alpha_co = alpha  # Actualizar alpha_co correctamente
            
            # Búsqueda Fina para optimizar alpha
            alpha_n, _ = quasi_newton_method(sa, alpha_co, frft, max_iterations, tol)  # Ajuste: No necesitamos u_n aquí

            # Corrección de Dispersión
            corrected_signal = apply_phase_correction(sa, alpha_n, omega_0)
            corrected_volume[:, x, y] = corrected_signal

    return corrected_volume

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
#%$
path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Data Boston\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'
file = '[p.Calibration][s.Mirror][02-10-2023_15-17-52].dispersion'
dispersion = np.fromfile(os.path.join(path,file))
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
fringes = ifft(tom,axis=0)
#%%

corrected_signal = apply_dispersion_correction(np.real(fringes), 
                                               frft, 
                                               omega_0=800, 
                                               alpha_step=0.5, 
                                               max_iterations=10, 
                                               tol=1e-6)
tom_corrected = fast_reconstruct(corrected_signal)
#%%
