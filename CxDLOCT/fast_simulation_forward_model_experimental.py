#%%
import numpy as np

# Tamaño de la malla
N = 10
L = 1.0  # Dimensión física de la muestra (en unidades arbitrarias)
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z)
grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Definir una función de susceptibilidad sintética
def chi(r):
    x, y, z = r
    return np.exp(-(x**2 + y**2 + z**2) / 0.1)

chi_values = np.array([chi(r) for r in grid_points]).reshape(N, N, N)
import scipy.constants as const

def A(k):
    k0 = 2 * np.pi / 0.83e-6  # Longitud de onda central (830 nm)
    sigma_k = 2 * np.pi / 0.34e-6  # Ancho de banda (340 nm)
    return np.exp(-0.5 * ((k - k0) / sigma_k)**2)

def G(r_prime, r0, k):
    distance = np.linalg.norm(r_prime - r0)
    return np.exp(1j * k * distance) / (4 * np.pi * distance)

def g(r_minus_r0, k, w0):
    z = r_minus_r0[2]
    r = np.linalg.norm(r_minus_r0[:2])
    z_R = np.pi * w0**2 / (2 * np.pi / k)  # Longitud de Rayleigh
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    R_z = z * (1 + (z_R / z)**2)
    phi_z = np.arctan(z / z_R)
    return (w0 / w_z) * np.exp(-r**2 / w_z**2) * np.exp(-1j * k * z) * np.exp(1j * (k * r**2 / (2 * R_z) - phi_z))

def K(r_prime, r0, k, w0):
    return A(k) * G(r_prime, r0, k) * g(r_prime - r0, k, w0) * g(r_prime - r0, k, w0)

def S(r0, k, chi_values, w0, grid_points):
    integral_result = 0
    for i, r_prime in enumerate(grid_points):
        integral_result += K(r_prime, r0, k, w0) * chi_values.ravel()[i]
    return integral_result

# Parámetros de la simulación
w0 = 0.01  # Tamaño de la cintura del haz (en unidades arbitrarias)
k_values = np.linspace(1e6, 5e6, 50)  # Números de onda (en unidades arbitrarias)
r0 = np.array([0, 0, 0])  # Posición transversal (centro de la muestra)

# Calcular la señal para cada k
S_values = np.array([S(r0, k, chi_values, w0, grid_points) for k in k_values])
# Nivel de ruido (relativo a la señal)
noise_level = 0.05
noise = noise_level * np.random.normal(size=S_values.shape)
S_noisy = S_values + noise
from scipy.linalg import lstsq

def construct_K_matrix(r0, k_values, w0, grid_points):
    K_matrix = []
    for k in k_values:
        row = [K(r_prime, r0, k, w0) for r_prime in grid_points]
        K_matrix.append(row)
    return np.array(K_matrix)

# Construir la matriz K
K_matrix = construct_K_matrix(r0, k_values, w0, grid_points)
S_vector = S_noisy

# Parámetro de regularización
alpha = 1e-2

# Regularización de Tikhonov
L = np.eye(K_matrix.shape[1])  # Operador de suavidad (identidad)
A = np.dot(K_matrix.T, K_matrix) + alpha * np.dot(L.T, L)
b = np.dot(K_matrix.T, S_vector)

# Resolver usando la pseudo-inversa
chi_hat = lstsq(A, b)[0]
chi_reconstructed = chi_hat.reshape(N, N, N)

# Visualización de los resultados
import matplotlib.pyplot as plt
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(chi_values[:, :, N//2], cmap='viridis')
axes[0].set_title("Distribución Original de $\\chi$")
axes[1].imshow(abs(chi_reconstructed[:, :, N//2]), cmap='viridis')
axes[1].set_title("Distribución Reconstruida de $\\chi$")
plt.show()
