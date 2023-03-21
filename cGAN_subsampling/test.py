# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# t = np.linspace(1,1000,10000)
# f = 1 * np.cos(2*np.pi*10*t)
# plt.plot(f)
# plt.show(block=True)
# print(len(t))

# # fig = px.line(f)
# # fig.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def simulate_michelson_interferometer(wavelength, path_difference, x_range, y_range, intensity=1):
#     X, Y = np.meshgrid(x_range, y_range)
#     k = 2 * np.pi / wavelength
    
#     beam1 = intensity * np.cos(k * X)
#     beam2 = intensity * np.cos(k * (X + path_difference))
    
#     interference_pattern = beam1 + beam2
#     return X, Y, interference_pattern

# wavelength = 20e-9  # longitud de onda en metros (500 nm)
# path_difference = 0.5 * wavelength  # diferencia de camino en metros (un ejemplo, puede variar)
# x_range = np.linspace(0, 20e-6, 1000)  # rango x en metros
# y_range = np.linspace(0, 20e-6, 1000)  # rango y en metros

# X, Y, interference_pattern = simulate_michelson_interferometer(wavelength, path_difference, x_range, y_range)

# plt.figure(figsize=(8, 8))
# plt.pcolormesh(X, Y, interference_pattern, cmap='viridis', shading='auto')
# plt.colorbar(label='Intensidad')
# plt.xlabel('Distancia en x (m)')
# plt.ylabel('Distancia en y (m)')
# plt.title('Patrón de interferencia de un interferómetro de Michelson')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# wavelength = 632.8e-9  # Wavelength of the light source (in meters)
# n_mirrors = 100  # Number of mirrors to simulate
# collision_broadening = 1e-12  # Collision broadening (in seconds)

# # Calculate the phase shift due to collision broadening
# phase_shift = 2 * np.pi * collision_broadening / wavelength

# # Create an array of mirror distances
# distances = np.linspace(0, n_mirrors * wavelength / 2, n_mirrors)

# # Calculate the interference pattern
# intensity = np.zeros(n_mirrors)
# for i in range(n_mirrors):
#     delta_phi = 4 * np.pi * distances[i] / wavelength
#     intensity[i] = (np.cos(delta_phi / 2 + phase_shift / 2)**2 *
#                     (1 + np.cos(phase_shift)))

# # Normalize the intensity
# intensity /= np.max(intensity)

# # Plot the interference pattern
# plt.plot(distances, intensity)
# plt.xlabel('Mirror Distance (m)')
# plt.ylabel('Normalized Intensity')
# plt.title('Collision-Broadened Light Source in a Michelson Interferometer')
# plt.grid(True)
# plt.show()

# import numpy as np

# def electric_field(t, E0, omega0, phi_k):
#     return E0 * np.exp(-1j * omega0 * t) * np.sum(np.exp(-1j * phi_k(t)), axis=0)
# def coherence(tau, E0, omega0, phi_k, t_samples):
#     E_t = electric_field(t_samples, E0, omega0, phi_k)
#     E_t_tau = electric_field(t_samples + tau, E0, omega0, phi_k)
    
#     numerator = np.mean(np.conj(E_t) * E_t_tau)
#     denominator = np.sqrt(np.mean(np.abs(E_t)**2) * np.mean(np.abs(E_t_tau)**2))
    
#     g1 = numerator / denominator
#     return g1
# E0 = 1.0
# omega0 = 2 * np.pi * 1e14  # Angular frequency of the light source (in radians per second)
# tau0 = 1e-12  # Collision broadening time constant (in seconds)

# # Phase shift function for the k-th collision
# def phi_k(t):
#     return np.random.uniform(-np.pi, np.pi, size=(len(t), 1000))

# t_samples = np.linspace(0, 1e-11, 1000)  # Time samples
# tau_values = np.linspace(-1e-12, 1e-12, 100)  # Range of τ values

# g1_values = [coherence(tau, E0, omega0, phi_k, t_samples) for tau in tau_values]
# g1_theoretical = np.exp(-1j * omega0 * tau_values - (tau_values / tau0))

# # Compare the simulated and theoretical coherence functions
# import matplotlib.pyplot as plt
# plt.plot(tau_values, np.real(g1_values), label='Simulated g₁(τ) - Real Part')
# plt.plot(tau_values, np.real(g1_theoretical), '--', label='Theoretical g₁(τ) - Real Part')
# plt.plot(tau_values, np.imag(g1_values), label='Simulated g₁(τ) - Imaginary Part')
# plt.plot(tau_values, np.imag(g1_theoretical), '--', label='Theoretical g₁(τ) - Imaginary Part')
# plt.xlabel('τ (s)')
# plt.ylabel('g₁(τ)')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 10.0  # Longitud de la cuerda
T = 2.0   # Duración de la simulación
Nx = 100  # Número de puntos de la malla en el espacio
Nt = 200  # Número de puntos de la malla en el tiempo
c = 1.0   # Velocidad de la onda

# Tamaño de los incrementos en espacio y tiempo
dx = L / (Nx - 1)
dt = T / (Nt - 1)

# Coeficiente de la ecuación
alpha = (c * dt / dx)**2

# Crear mallas de espacio y tiempo
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Inicializar soluciones
u = np.zeros((Nx, Nt))

# Condición inicial
u[:, 0] = np.sin(np.pi * x / L)  # Función seno como ejemplo
u[:, 1] = u[:, 0] + 0.5 * alpha * (np.roll(u[:, 0], -1) - 2 * u[:, 0] + np.roll(u[:, 0], 1))

# Resolver la ecuación de onda utilizando diferencias finitas
for n in range(1, Nt - 1):
    u[:, n + 1] = 2 * (1 - alpha) * u[:, n] - u[:, n - 1] + alpha * (np.roll(u[:, n], -1) - 2 * u[:, n] + np.roll(u[:, n], 1))

# Graficar solución
plt.figure()
plt.imshow(u, extent=[0, T, 0, L], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Amplitud')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Ecuación de onda unidimensional resuelta con diferencias finitas')
plt.show()

