import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tamaño de la imagen 3D
width, height, depth = 100, 100, 100

# Creamos la matriz 3D que representará el patrón de interferencia
image = np.zeros((width, height, depth))

# Asignamos fase cero al espacio libre (todo el volumen inicialmente)
phase = np.zeros((width, height, depth))

# Definimos las capas con sus respectivas densidades (número de partículas), índices de refracción y coeficientes de absorción
layers = [(20, 5000, 1.1, 0.05), (40, 10000, 1.2, 0.1), (50, 15000, 1.3, 0.15), (70, 10000, 1.2, 0.05), (90, 5000, 1.1, 0.1)]  # (posición inicial z, densidad, índice de refracción, coeficiente de absorción)

thickness = 10  # grosor de las capas

for start_z, density, ref_index, absorption in layers:
    for z in range(start_z, start_z + thickness):  # iteramos a través de varias posiciones z para cada capa
        # Selección aleatoria de posiciones (x, y) para las partículas en cada capa
        particles_x = np.random.randint(0, width, density)
        particles_y = np.random.randint(0, height, density)

        # Asignación de fase aleatoria a cada partícula, modificada por el índice de refracción
        phase[particles_x, particles_y, z] = np.random.uniform(0, 2*np.pi, density) * ref_index

    # Modificamos la intensidad de la luz en la capa de acuerdo con la ley de Beer-Lambert
    intensity = np.exp(-absorption * thickness)

    # Aplicamos la absorción a la fase de la luz en la capa
    phase[start_z:start_z + thickness] *= intensity

# Procedemos a calcular la intensidad basada en la diferencia de fase con el haz de referencia
reference_phase = 0
phase_difference = phase - reference_phase
intensity = np.cos(phase_difference)**2
image = (intensity / intensity.max()) * 255

# Invertimos los colores restando la imagen de la intensidad máxima
image = image.max() - image

# Creamos una figura 3D para visualizar la matriz 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Creamos una malla de puntos para visualizar la matriz 3D
x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))

# Ahora vamos a visualizar solo los puntos donde la intensidad es menor a un cierto umbral
threshold = 55
ax.scatter(x[image < threshold], y[image < threshold], z[image < threshold], c=image[image < threshold], cmap='gray', s=5)

plt.show()








