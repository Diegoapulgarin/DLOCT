#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from numpy.fft import fft,fftshift
import matplotlib.animation as animation
from tqdm import tqdm
#%%
# Parámetros
lambda_0 = 1.3e-6  # Longitud de onda central en metros
delta_lambda = 100e-9  # Ancho de banda en metros
c = 3e8  # Velocidad de la luz en m/s
k_0 = 2 * np.pi / lambda_0  # Número de onda central
delta_k = 2 * np.pi * delta_lambda / lambda_0**2  # Ancho de banda en número de onda
f_0 = c / lambda_0  # Frecuencia central
num_periods = 100  # Número de periodos para muestrear
fs = num_periods * (f_0+delta_lambda)  # Frecuencia de muestreo (al menos 10 veces la frecuencia central)
T = num_periods / f_0  # Tiempo total de muestreo
num_points = int(T * fs)  # Número de puntos de muestreo
t = np.linspace(0, T, num_points)  # Vector de tiempo

# Posiciones de los reflectores en metros
z_R = 0  # Posición del reflector de referencia
z_n = np.array([0.5e-2, 1e-2, -2e-2])  # Posiciones de los reflectores de muestra

# Reflectividades
R_R = 1.0  # Reflectividad del reflector de referencia
R_n = np.array([0.8, 0.7, 0.3])  # Reflectividades de los reflectores de muestra

omega_D_vector = 2 * np.pi * c/(lambda_0/np.linspace(1,16,16))
# Parámetro de batimiento
omega_D = 2 * np.pi * c/(lambda_0/1)  # Frecuencia de batimiento en rad/s

# Variación temporal del número de onda
dk_dt = delta_k / T


aline_reconstructed = []
for omega_D in tqdm(omega_D_vector):
    # Generar la señal interferométrica con todos los términos
    i_t = np.zeros_like(t)
    for n in range(len(z_n)):
        omega_n = dk_dt * (z_R - z_n[n])
        phi_n = k_0* (z_n[n] - z_R)
        i_t += 2 * np.sqrt(R_R * R_n[n]) * np.cos((omega_n + omega_D) * t + phi_n)

        # Términos de autocorrelación
        i_t += R_n[n]


    # Términos cruzados de interferencia
    for n in range(len(z_n)):
        for m in range(len(z_n)):
            if n != m:
                omega_nm = dk_dt * (z_n[n] - z_n[m])
                phi_nm = k_0 * (z_n[n] - z_n[m])
                i_t += 0.2 * np.sqrt(R_n[n] * R_n[m]) * np.cos(omega_nm * t + phi_nm)


    i_fft = np.fft.fft(i_t)
    magnitude = np.abs(i_fft)
    aline_reconstructed.append(magnitude)
frequencies_fft = np.fft.fftfreq(t.size, d=1/fs)
alines = np.array(aline_reconstructed)
# %%
# Crear la figura y el eje
fig, ax = plt.subplots()
line2, = ax.plot(frequencies_fft, alines[0], label='FFT de la Señal Interferométrica Completa con Batimiento')
ax.set_xlim([frequencies_fft.min(), frequencies_fft.max()])
ax.set_ylim([0, np.max(alines)])
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Magnitud')
ax.legend()

# Función de actualización para la animación
def update(frame):
    line2.set_ydata(alines[frame])
    return line2,

# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=len(alines), interval=5)
ani.save(filename=r"C:\Users\USER\Documents\animaciones\modulation.html", writer="html")

# Guardar la animación
ani.save(filename=r"C:\Users\USER\Documents\animaciones\modulation.gif", writer="pillow",dpi=150)



