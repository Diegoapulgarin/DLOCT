#%%
'''
se supone una fuente de barrido con una longitud de onda central de 1.3 μm 
y una anchura de banda de 100 nm. Utilizaremos tres puntos reflectantes ubicados 
a 0.5 mm, 1 mm y 2 mm. Nuestro objetivo será generar la señal interferométrica, 
realizar la multiplicación por las señales del oscilador local y visualizar las 
componentes en fase e imaginaria.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from numpy.fft import fft,fftshift

#%%
# Parámetros
lambda_0 = 1.3e-6  # Longitud de onda central en metros
delta_lambda = 100e-9  # Ancho de banda en metros
c = 3e8  # Velocidad de la luz en m/s
k_0 = 2 * np.pi / lambda_0  # Número de onda central
delta_k = 2 * np.pi * delta_lambda / lambda_0**2  # Ancho de banda en número de onda
f_0 = c / lambda_0  # Frecuencia central
num_periods = 100  # Número de periodos para muestrear
fs = 100 * f_0  # Frecuencia de muestreo (al menos 10 veces la frecuencia central)
T = num_periods / f_0  # Tiempo total de muestreo
num_points = int(T * fs)  # Número de puntos de muestreo
t = np.linspace(0, T, num_points)  # Vector de tiempo

# Posiciones de los reflectores en metros
z_R = 0  # Posición del reflector de referencia
z_n = np.array([0.5e-2, 1e-2, -2e-2])  # Posiciones de los reflectores de muestra

# Reflectividades
R_R = 1.0  # Reflectividad del reflector de referencia
R_n = np.array([0.8, 0.7, 0.3])  # Reflectividades de los reflectores de muestra

# Parámetro de batimiento
omega_D = 2 * np.pi * c/(lambda_0/39)  # Frecuencia de batimiento en rad/s

# Variación temporal del número de onda
dk_dt = delta_k / T

# Generar la señal interferométrica con todos los términos
i_t = np.zeros_like(t)
for n in range(len(z_n)):
    omega_n = dk_dt * (z_R - z_n[n])
    phi_n = k_0* (z_n[n] - z_R)
    i_t += 2 * np.sqrt(R_R * R_n[n]) * np.cos((omega_n + omega_D) * t + phi_n)

    # Términos de autocorrelación
    i_t += R_n[n]

i_t2 = np.zeros_like(t)
for n in range(len(z_n)):
    omega_n = dk_dt * (z_R - z_n[n])
    phi_n = k_0* (z_n[n] - z_R)
    i_t2 += 2 * np.sqrt(R_R * R_n[n]) * np.cos((omega_n) * t + phi_n)

    # Términos de autocorrelación
    i_t2 += R_n[n]

# Términos cruzados de interferencia
# for n in range(len(z_n)):
#     for m in range(len(z_n)):
#         if n != m:
#             omega_nm = dk_dt * (z_n[n] - z_n[m])
#             phi_nm = k_0 * (z_n[n] - z_n[m])
#             i_t += 2 * np.sqrt(R_n[n] * R_n[m]) * np.cos(omega_nm * t + phi_nm)

plt.figure(figsize=(10, 4))
plt.plot(t, i_t, label='Señal Interferométrica Completa con Batimiento')
plt.xlabel('Tiempo (s)')
plt.ylabel('Intensidad')
plt.title('Señal Interferométrica Completa con Frecuencia de Batimiento Antes de Demodulación')
plt.legend()
plt.show()

# Transformada de Fourier de la señal interferométrica
i_fft = np.fft.fft(i_t)
frequencies_fft = np.fft.fftfreq(t.size, d=1/fs)
magnitude = np.abs(i_fft)

plt.figure(figsize=(10, 4))
plt.plot(frequencies_fft, magnitude, label='FFT de la Señal Interferométrica Completa con Batimiento')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Transformada de Fourier Antes de Demodulación')
plt.legend()
plt.show()

i_fft2 = np.fft.fft(i_t2)
frequencies_fft = np.fft.fftfreq(t.size, d=1/fs)
magnitude2 = np.abs(i_fft2)

plt.figure(figsize=(10, 4))
plt.plot(frequencies_fft, magnitude2, label='FFT de la Señal Interferométrica Completa con Batimiento')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Transformada de Fourier sin modular')
plt.legend()
plt.show()
# Demodulación
# Multiplicación por los osciladores locales
i_Re = i_t * np.cos(omega_D * t)
i_Im = i_t * np.sin(omega_D * t)

# Diseño de un filtro de paso de banda (Butterworth)
lowcut = omega_D-(omega_D*0.99)
highcut = omega_D+(omega_D*1e-16)
nyq = 2*fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(1, [low, high], btype='band')

# Aplicación del filtro a las componentes real e imaginaria por separado
i_Re_filtered = filtfilt(b, a, i_Re)
i_Im_filtered = filtfilt(b, a, i_Im)

# Visualización de las señales demoduladas y filtradas
plt.figure(figsize=(10, 4))
plt.plot(t, i_Re_filtered, label='Componente Real (Fase)')
plt.plot(t, i_Im_filtered, label='Componente Imaginaria (Cuadratura)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Intensidad')
plt.title('Componentes Demoduladas de la Señal Interferométrica')
plt.legend()
plt.show()

# Combinación de las señales filtradas en una señal compleja
i_complex = i_Re_filtered + 1j * i_Im_filtered

# Transformada de Fourier de la señal compleja
i_fft_filtered = np.fft.fft(i_complex)
frequencies_filtered = np.fft.fftfreq(t.size, d=t[1] - t[0])
magnitude_filtered = np.abs(i_fft_filtered)

plt.figure(figsize=(10, 4))
plt.plot(frequencies_filtered, magnitude_filtered, label='FFT de la Señal Demodulada y Filtrada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Transformada de Fourier Después de Demodulación y Filtrado')
plt.legend()
plt.show()

i_fft2 = np.fft.fft(i_t2)
frequencies_fft = np.fft.fftfreq(t.size, d=1/fs)
magnitude2 = np.abs(i_fft2)

plt.figure(figsize=(10, 4))
plt.plot(frequencies_fft, magnitude2, label='FFT de la Señal Interferométrica Completa con Batimiento')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Transformada de Fourier sin modular')
plt.legend()
plt.show()