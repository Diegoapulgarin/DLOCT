#%%
import numpy as np
from numpy.fft import fft,ifft,fftshift
import os
import matplotlib.pyplot as plt
from numpy.linalg import norm
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
     
#%%
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
fringes = fftshift(ifft(fftshift(tom,axes=0),axis=0),axes=0)
del tom
#%%
dispersion_coefficients = {
    2: -1.0e-4,  # Coeficiente para el término cuadrático
    3: 3.0e-6,   # Coeficiente para el término cúbico
}
lambda_c = 870e-9  # Longitud de onda central en metros
delta_lambda_sweep = 95e-9  # Rango de barrido en metros

# Calcular longitudes de onda mínima y máxima
lambda_min = lambda_c - (delta_lambda_sweep / 2)
lambda_max = lambda_c + (delta_lambda_sweep / 2)

# Generar un vector de longitudes de onda con distribución uniforme
num_points = 1024  # Número de puntos en el vector
lambda_vector = np.linspace(lambda_min, lambda_max, num_points)

# Convertir el vector de longitudes de onda a frecuencias angulares
omega_vector = 2 * np.pi * (3e8 / lambda_vector)

z_dim = fringes.shape[0] 
# Convertir longitudes de onda a números de onda (k_min y k_max)
k_min = 2 * np.pi / lambda_max  # Usamos lambda_max para k_min debido a la inversión de la relación
k_max = 2 * np.pi / lambda_min  # Usamos lambda_min para k_max

lambda_0_nm = 870
lambda_0_m = lambda_0_nm * 1e-9
omega_0 = 2 * np.pi * (3e8 / lambda_0_m)
d = 1.9e-3
#%%

def apply_defr_corrected(volume, omega_vector, dispersion_coefficients, omega_0, d, max_iterations=10, tolerance=1e-6):
    """
    Aplica el algoritmo DEFR para compensación de dispersión en datos OCT, ajustado según el modelo matemático.
    
    Parameters:
    - volume: Volumen de datos OCT con forma (z, x, y, channels).
    - omega_vector: Vector de frecuencias ópticas.
    - dispersion_coefficients: Coeficientes de dispersión ajustados {ai}.
    - omega_0: Frecuencia central óptica.
    - d: Espesor de la muestra o distancia sobre la cual se calcula la dispersión.
    - max_iterations: Número máximo de iteraciones.
    - tolerance: Umbral para el criterio de parada basado en la energía residual.
    
    Returns:
    - volume_compensated: Volumen de datos OCT compensado.
    """
    volume_compensated = volume.copy()  # Inicializar con el volumen original
    prev_volume = volume.copy()
    
    # Calcular la fase compensatoria basada en la expansión de Taylor
    phi_omega = np.zeros_like(omega_vector, dtype=complex)
    for order, coefficient in dispersion_coefficients.items():
        if order > 1:  # Ignorar los primeros dos términos (orden 0 y 1) que no contribuyen a la fase
            phi_omega += (coefficient * (2*d) * (omega_vector - omega_0)**order)

    for iteration in range(max_iterations):
        # Transformada de Fourier
        volume_f = fft(volume_compensated, axis=0)
        
        # Aplicar la fase compensatoria
        phase_adjustment = np.exp(-1j * phi_omega)
        volume_f_adjusted = volume_f * phase_adjustment.reshape(-1, 1, 1, 1)
        
        # Transformada Inversa de Fourier para aplicar el ajuste
        volume_compensated = ifft(volume_f_adjusted, axis=0)
        
        # Calcular la diferencia (energía residual) entre iteraciones
        residual = norm(volume_compensated - prev_volume)
        print('iteration:', iteration, 'residual:', residual)
        if residual < tolerance:
            print(f"Convergencia alcanzada en la iteración {iteration}")
            break
        
        prev_volume = volume_compensated.copy()
        
    return volume_compensated


final_fringes = apply_defr_corrected(fringes[:,:,0], omega_vector, dispersion_coefficients, omega_0, d, max_iterations=10, tolerance=1e-6)
tom = fftshift(fft(fftshift(final_fringes, axes=0), axis=0), axes=0)     
plt.imshow(10*np.log10(abs(tom[:,:,0])**2))
