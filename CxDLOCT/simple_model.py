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
layers = [(20, 10000, 1.1, 0.01), (40, 10000, 1.2, 0.01), (50, 15000, 1.3, 0.05), (75, 10000, 1.2, 0.03), (90, 5000, 1.1, 0)]  # (posición inicial z, densidad, índice de refracción, coeficiente de absorción)

thickness = 10  # grosor de las capas

# Comenzamos con una intensidad de luz de 1 para el haz de referencia y el haz reflejado
intensity1 = 70.0
intensity2 = intensity1

for start_z, density, ref_index, absorption in layers:
    for z in range(start_z, start_z + thickness):
        particles_x = np.random.randint(0, width, density)
        particles_y = np.random.randint(0, height, density)

        # Disminuimos la intensidad de la luz según la ley de Beer-Lambert
        intensity2 = intensity2 * np.exp(-absorption * (z - start_z + 1))
        # print(intensity2)

        # Ahora, el cambio de fase depende de la intensidad de la luz
        phase[particles_x, particles_y, z] += np.random.uniform(0, 2*np.pi, density) * ref_index * intensity2



    # Modificamos la intensidad de la luz reflejada en la capa de acuerdo con la ley de Beer-Lambert
    intensity2 *= np.exp(-absorption * thickness)

# Procedemos a calcular la intensidad basada en la diferencia de fase con el haz de referencia
reference_phase = 0
phase_difference = phase - reference_phase
intensity = intensity1 + intensity2 + 2*np.sqrt(intensity1*intensity2)*np.cos(phase_difference)
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




import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Coordenada específica para las vistas ortogonales
posicion = 99

# Obtener las vistas ortogonales en la coordenada específica
vista_xy = image[:, :, posicion]  # Vista XY (proyección en el plano Z)
vista_xz = image[:, posicion, :]  # Vista XZ (proyección en el plano Y)
vista_yz = image[posicion, :, :]  # Vista YZ (proyección en el plano X)

fig = make_subplots(rows=1, cols=3, subplot_titles=('Vista XY', 'Vista XZ', 'Vista YZ'))
fig.add_trace(go.Heatmap(z=vista_xy, colorscale='Viridis'), row=1, col=1)
fig.add_trace(go.Heatmap(z=vista_xz, colorscale='Viridis'), row=1, col=2)
fig.add_trace(go.Heatmap(z=vista_yz, colorscale='Viridis'), row=1, col=3)

# Personalizar el diseño del gráfico
fig.update_layout(
    title='Vistas Ortogonales 2D',
    height=600,
    width=1000
)

# Mostrar el gráfico
fig.show()
# fig2 = px.line(intensity[50,50,:])
# fig2.show()