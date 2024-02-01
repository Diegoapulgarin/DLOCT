#%%
import numpy as np
from numpy.fft import fft,ifft,fftshift
import os
import matplotlib.pyplot as plt

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
def dispersion_compensation(volume, k_vector, dispersion_coefficients):
    """
    Applies dispersion compensation to an OCT volume data.
    Parameters:
    - volume: numpy array with shape (z, x, y, channels), OCT data.
    - k_vector: numpy array, linearized wave number vector.
    - dispersion_coefficients: dict, dispersion coefficients of the system.
    Returns:
    - compensated_volume: numpy array, volume with dispersion compensated.
    """
    # Apply FFT along the z-axis
    volume_f = fft(volume, axis=0)
    # Calculate the compensation phase
    compensation_phase = np.zeros_like(k_vector)
    for order, coefficient in dispersion_coefficients.items():
        compensation_phase += coefficient * k_vector**order
    # Adjust the shape of compensation_phase to match volume_f for broadcasting
    compensation_phase = np.exp(-1j * compensation_phase)
    compensation_phase = compensation_phase.reshape(-1, 1, 1, 1)
    # Check if compensation_phase length matches the z-axis of volume_f
    if compensation_phase.shape[0] != volume_f.shape[0]:
        raise ValueError("The length of compensation_phase does not match the z-axis of the volume_f. Adjustment required.")
    # Apply the phase compensation
    compensated_volume_f = volume_f * compensation_phase
    # Apply IFFT to return to the time domain
    compensated_volume = ifft(compensated_volume_f, axis=0)
    return compensated_volume
coeficientes_dispersion = {
    2: -1.0e-4,  # Coeficiente para el término cuadrático
    3: 3.0e-6,   # Coeficiente para el término cúbico
}
lambda_c = 1300e-9  # Longitud de onda central en metros
delta_lambda_sweep = 95e-9  # Rango de barrido en metros

# Calcular longitudes de onda mínima y máxima
lambda_min = lambda_c - (delta_lambda_sweep / 2)
lambda_max = lambda_c + (delta_lambda_sweep / 2)
z_dim = fringes.shape[0] 
# Convertir longitudes de onda a números de onda (k_min y k_max)
k_min = 2 * np.pi / lambda_max  # Usamos lambda_max para k_min debido a la inversión de la relación
k_max = 2 * np.pi / lambda_min  # Usamos lambda_min para k_max

# Generar el vector k
indices = np.arange(1, 1281, 2)  # Basado en el mapeo proporcionado
k = np.linspace(k_min, k_max, z_dim)

compensedFringes =dispersion_compensation(fringes,k,coeficientes_dispersion)
#%%
# Usar estos coeficientes en el algoritmo DEFR
factor = 1
# Modificar los coeficientes de dispersión para exagerar el ajuste
coeficientes_dispersion_adjusted = {order: coef * factor for order, coef in coeficientes_dispersion.items()}

def apply_defr(volume, k_vector, dispersion_coefficients, max_iterations=10, tolerance=1e-6):
    """
    Aplica el algoritmo DEFR para compensación de dispersión en datos OCT.
    
    Parameters:
    - volume: Volumen de datos OCT con forma (z, x, y, channels).
    - k_vector: Vector de números de onda linealizados.
    - dispersion_coefficients: Coeficientes de dispersión.
    - max_iterations: Número máximo de iteraciones.
    - tolerance: Umbral para el criterio de parada basado en la energía residual.
    
    Returns:
    - volume_compensated: Volumen de datos OCT compensado.
    """
    volume_compensated = volume.copy()  # Inicializar con el volumen original
    prev_volume = volume.copy()
    
    for iteration in range(max_iterations):
        
        # Transformada de Fourier
        volume_f = fft(volume_compensated, axis=0)
        
        # Ajuste específico de compensación de dispersión
        phase_adjustment = np.exp(-1j * np.sum([coefficient * k_vector**order for order, coefficient in dispersion_coefficients.items()], axis=0))
        volume_f_adjusted = volume_f * phase_adjustment.reshape(-1, 1, 1, 1)
        
        # Transformada Inversa de Fourier para aplicar el ajuste
        volume_compensated = ifft(volume_f_adjusted, axis=0)
        
        # Calcular la diferencia (energía residual) entre iteraciones
        residual = norm(volume_compensated - prev_volume)
        print('iteration:',iteration,'residual:',residual)
        if residual < tolerance:
            print(f"Convergencia alcanzada en la iteración {iteration}")
            break
        
        prev_volume = volume_compensated.copy()
        
    return volume_compensated


final_fringes = apply_defr(compensedFringes,k,coeficientes_dispersion_adjusted,max_iterations=10,tolerance=1e2)
tom = fftshift(fft(fftshift(final_fringes, axes=0), axis=0), axes=0)     
plt.imshow(10*np.log10(abs(tom[:,:,0,0])**2))
