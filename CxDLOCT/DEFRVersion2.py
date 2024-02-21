#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
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

def plot(bscan,zmax,zmin):
     fig = px.imshow(10*np.log10(abs(bscan)**2),color_continuous_scale='viridis',zmax=zmax,zmin=zmin)
     fig.show()

def identify_max_component_aline(aline):
    """
    Identifica el componente máximo en la A-line dada.
    
    :param aline: La A-line (una dimensión de ci1 o ci2) como un array de NumPy.
    :return: índice y valor del componente máximo en la A-line.
    """
    max_value = np.max(aline)
    max_index = np.argmax(aline)
    return max_index, max_value

def update_outputs_and_subtract_artifacts_aline(d_aline, ci_aline, max_component_index):
    """
    Actualiza la A-line basándose en la eliminación de artefactos.
    
    :param d_aline: Salida acumulativa para la A-line de ci.
    :param ci_aline: A-line del espectro compensado por dispersión o no compensado.
    :param max_component_index: Índice del componente máximo en la A-line.
    :return: La A-line actualizada.
    """
    # Sustraer el componente máximo de la A-line
    ci_aline[max_component_index] = 0  # Asumiendo que queremos eliminar ese pico por completo
    d_aline[max_component_index] = 0  # Actualizar la salida acumulativa también
    
    return d_aline, ci_aline

#%%
path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Data Boston\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'
file = '[p.Calibration][s.Mirror][02-10-2023_15-17-52].dispersion'
dispersion = np.fromfile(os.path.join(path,file))
# plt.plot(dispersion)
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
tom2 = fft(fringes,axis=0)
#%%

# Calcular el número de puntos en el dominio k
num_puntos_k = len(dispersion)

# Crear un nuevo array k para el doble de puntos
k_nuevo = np.linspace(-1, 1, 2 * num_puntos_k)

# Interpolar la dispersión al nuevo dominio k
dispersion_interpolada = np.interp(k_nuevo, np.linspace(-np.pi, np.pi, num_puntos_k), dispersion)



fftDispersion = fft(dispersion_interpolada)[0:int(len(dispersion_interpolada)/2)]
n = len(fftDispersion)
# full_spectrum = np.zeros(len(fringes), dtype=complex)
# full_spectrum[0:n] = fftDispersion  # Ignoramos el primer elemento por ahora
# phasedispersion_resampled = np.angle(full_spectrum)
phasedispersion = np.angle(ifft(fftDispersion))
unwrapped_phase = np.unwrap(phasedispersion)
coeficientes = np.polyfit(np.arange(len(unwrapped_phase)), unwrapped_phase, 4)
polinomio = np.poly1d(coeficientes)
fase_lineal = polinomio(np.arange(len(unwrapped_phase)))
dispersivePhase = unwrapped_phase - fase_lineal

inverseFourierDispersive = fftshift(ifft(np.exp(-2j*dispersivePhase)))
# plt.plot(abs(inverseFourierDispersive))
# # Parámetros para el modelo de dispersión (ajustar según sea necesario)
# B1 = 1.0  # Ejemplo de coeficiente para el modelo
# C1 = 1.0e-14  # Ejemplo de coeficiente (asegúrate de que esté en las unidades correctas)
# lambda_0 = 800e-9  # Longitud de onda central en metros

# # Rango de números de onda k (inverso de la longitud de onda, asumiendo un espectro óptico)
# k_min = 2 * np.pi / (lambda_0 + 100e-9)  # Número de onda mínimo
# k_max = 2 * np.pi / (lambda_0 - 100e-9)  # Número de onda máximo
# k = np.linspace(k_min, k_max, 2304)  # Vector de números de onda

# # Conversión de número de onda a longitud de onda para usar en la ecuación de Sellmeier
# lambda_k = 2 * np.pi / k
# n_k = np.sqrt(1 + B1 * lambda_k**2 / (lambda_k**2 - C1))  # Índice de refracción
# phi_k = (n_k - 1) * k  # Fase de dispersión, asumiendo una longitud de referencia
newfringes = ifft(fft(fringes,axis=0),axis=0)
dispersion_compensation = np.exp(-1j * dispersivePhase)
negative_phase_correction = np.exp(-1j * dispersivePhase)[:, np.newaxis, np.newaxis]
c1 = ifft(newfringes * negative_phase_correction, axis=0)
c2 = ifft(newfringes, axis=0)
#%%
# Aplicar la función de fase negativa y realizar la IFFT para obtener c1(n)
# negative_phase_correction = np.exp(-1j * dispersion)[:, np.newaxis, np.newaxis]
c1 = ifft(fringes * negative_phase_correction, axis=0)
c2 = ifft(fringes, axis=0)
positive_phase_correction = np.exp(1j * dispersion)
negative_phase_correction = np.exp(-1j * dispersion)
double_negative_phase_correction = np.exp(-2j * dispersion)
p1p = ifft(positive_phase_correction)
p1n = ifft(negative_phase_correction)
p2 =  ifft(double_negative_phase_correction)

#%%
# plot(c1[500:1600,:,0]-c2[500:1600,:,0])
M = 10  # Número máximo de iteraciones definido
i = 0  # Índice de iteración inicial
# Variables para acumular los resultados
d1 = np.zeros_like(c1)  # Inicializar d1
d2 = np.zeros_like(c2)  # Inicializar d2
# Asumimos que 'N' es el número de muestras en la dirección z
N = c1.shape[0]

while i < M:
    for x in range(c1.shape[1]):
        for y in range(c1.shape[2]):
            # Identificar los índices de los máximos
            n1_i = np.argmax(np.abs(c1[:, x, y]))
            n2_i = np.argmax(np.abs(c2[:, x, y]))
            
            # Corrección pixel a pixel
            for n in range(N):
                # Compara los picos y actualiza los espectros y salidas
                if np.abs(c1[n1_i, x, y]) > np.abs(c2[n2_i, x, y]):
                    peak_value = c1[n1_i, x, y] - (c1[n1_i, x, y].conj() * p2[(n + n1_i) % N])
                    d1[n, x, y] += c1[n1_i, x, y]  # Añadir el pico a d1[n]
                    c1[n, x, y] -= peak_value  # Sustraer el pico corregido de c1[n]
                    # Corrección de c2[n] dependiendo de la posición de n1_i
                    if n1_i <= N//2:
                        corrected_index = (n - n1_i) % N
                        c2[n, x, y] -= peak_value * p1p[corrected_index]
                    else:
                        corrected_index = (n - n1_i + N) % N
                        c2[n, x, y] -= peak_value.conj() * p1n[corrected_index]
                else:
                    peak_value = c2[n2_i, x, y] - (c2[n2_i, x, y].conj() * p1n[(n + n2_i) % N])
                    d2[n, x, y] += c2[n2_i, x, y]  # Añadir el pico a d2[n]
                    c2[n, x, y] -= peak_value  # Sustraer el pico corregido de c2[n]
                    # Aseguramos que el índice esté dentro de los límites del array p1p
                    corrected_index = (n - n2_i + N) % N
                    c1[n, x, y] -= peak_value.conj() * p1p[corrected_index]  # Corrección de c1[n]
    
    i += 1
    if i < M:
        continue  # Volver al paso 2 si no se ha alcanzado el número máximo de iteraciones
    else:
        d1 += c1  # Añadir el espectro restante de c1 a d1
        d2 += c2  # Añadir el espectro restante de c2 a d2
        break  # Salir del bucle si se alcanza el número máximo de iteraciones

#%%
plot(c1[:,:,0])


