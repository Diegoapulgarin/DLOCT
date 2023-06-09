#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import hilbert
#%%
# Tamaño de la imagen 3D
lamb = 633*1e-9
dx = 5.2*1e-6
n = 100
width, height, depth = n, n, n
C=n/2
R=n/2
diff = 25
Cmax = n/2 - diff
Rmax = n/2 - diff
k = 2*np.pi/lamb
ThetaXM=np.arcsin((C-Cmax)*lamb/(n*dx))
ThetaYM=np.arcsin((R-Rmax)*lamb/(n*dx))
x = np.linspace(-n/2,n/2,n)
y = np.linspace(-n/2,n/2,n)
X,Y = np.meshgrid(x,y)
R0=np.exp(1j*k*(np.sin(ThetaXM)*X*dx+np.sin(ThetaYM)*Y*dx))
plt.imshow(np.angle(R0))
#%%


# Creamos la matriz 3D que representará el patrón de interferencia
image = np.zeros((width, height, depth))

# Asignamos fase cero al espacio libre (todo el volumen inicialmente)
phase = np.zeros((width, height, depth))+ np.angle(R0)

# Definimos las capas con sus respectivas densidades (número de partículas), índices de refracción y coeficientes de absorción
layers = [(20, 10000, 1.1, 0.03), (40, 10000, 1.2, 0.05), (50, 1500, 1.3, 0.01), (75, 10000, 1.2, 0.03), (90, 50000, 1.1, 0)]  # (posición inicial z, densidad, índice de refracción, coeficiente de absorción)

thickness = 10  # grosor de las capas

# Comenzamos con una intensidad de luz de 1 para el haz de referencia y el haz reflejado
intensity1 = 100.0
intensity2 = intensity1

for start_z, density, ref_index, absorption in layers:
    for z in range(start_z, start_z + thickness):
        particles_x = np.random.randint(0, width, density)
        particles_y = np.random.randint(0, height, density)
        # print(z)
        # Disminuimos la intensidad de la luz según la ley de Beer-Lambert
        # intensity2 = intensity2 * np.exp(-absorption * (z - start_z + 1)*thickness)
        print(intensity2)

        # Ahora, el cambio de fase depende de la intensidad de la luz
        phase[particles_x, particles_y, z] += np.random.uniform(0, 2*np.pi, density) * ref_index * intensity2
        intensity2 = intensity2 * np.exp(-absorption * (z - start_z ))
    # Modificamos la intensidad de la luz reflejada en la capa de acuerdo con la ley de Beer-Lambert
    # intensity2 *= np.exp(-absorption * thickness)

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
posicion = 95

# Obtener las vistas ortogonales en la coordenada específica
vista_xy = image[:, :, posicion]  # Vista XY (proyección en el plano Z)
vista_xz = image[:, posicion, :]  # Vista XZ (proyección en el plano Y)
vista_yz = image[posicion, :, :]  # Vista YZ (proyección en el plano X)

fig = make_subplots(rows=1, cols=2, subplot_titles=('Vista XY', 'Vista XZ'))
fig.add_trace(go.Heatmap(z=vista_xy, colorscale='Viridis'), row=1, col=1)
fig.add_trace(go.Heatmap(z=vista_xz, colorscale='Viridis'), row=1, col=2)
# fig.add_trace(go.Heatmap(z=vista_yz, colorscale='Viridis'), row=1, col=3)
fig.update_xaxes(title='Pixels')
fig.update_yaxes(title='Pixels')

fig.update_layout(title={
        'text': 'Orthogonal Views',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24}},
        font_family="Raleway",
        height=600,
        width=1000)

# Mostrar el gráfico
fig.show()
fig.write_html("othogonal view.html")
#%%

# Altura antes y después de las capas
before_layer_z = 5
after_layer_z = 95

before_layer_interference = abs((R0+(intensity[:, :, before_layer_z]))**2)
after_layer_interference = abs((R0+(intensity[:, :, after_layer_z]))**2)
before_layer_interference = before_layer_interference/np.max(before_layer_interference)*255
after_layer_interference = after_layer_interference/np.max(after_layer_interference)*255
profile_before = np.mean(before_layer_interference,axis=0)
profile_after = np.mean(after_layer_interference,axis=0)
analytic_signal = hilbert(profile_before)
envelope = np.abs(analytic_signal)

# # Calculamos la diferencia de fase en cada altura y obtenemos la interferencia
# before_layer_phase_difference = phase[:, :, before_layer_z] - reference_phase
# before_layer_interference = reference_intensity + intensity[:, :, before_layer_z] + 2*np.sqrt(reference_intensity*intensity[:, :, before_layer_z])*np.cos(before_layer_phase_difference)

# after_layer_phase_difference = phase[:, :, after_layer_z] - reference_phase
# after_layer_interference = reference_intensity + intensity[:, :, after_layer_z] + 2*np.sqrt(reference_intensity*intensity[:, :, after_layer_z])*np.cos(after_layer_phase_difference)
#%%
fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Before first layer', 'after last layer'))
fig2.add_trace(go.Heatmap(z=before_layer_interference, colorscale='Viridis'), row=1, col=1)
fig2.add_trace(go.Heatmap(z=after_layer_interference, colorscale='Viridis'), row=1, col=2)
# fig.add_trace(go.Heatmap(z=vista_yz, colorscale='Viridis'), row=1, col=3)
fig2.update_xaxes(title='Pixels')
fig2.update_yaxes(title='Pixels')
fig2.update_layout(title={
        'text': 'Interferometric comparision',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24}},
        font_family="Raleway",
        height=600,
        width=1000)
fig2.show()
fig2.write_html("interferometric_comparision.html")
#%%
fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Before first layer', 'after last layer'))
fig3.add_trace(
    go.Scatter(y=profile_before, mode='lines', name='Before'),
    row=1, col=1)
fig3.add_trace(
    go.Scatter(y=profile_after, mode='lines', name='After'),
    row=1, col=2)
# fig.add_trace(go.Heatmap(z=vista_yz, colorscale='Viridis'), row=1, col=3)
fig.update_yaxes(title_text="Intensity", row=1, col=1)
fig.update_yaxes(title_text="Intensity", row=1, col=2)
fig.update_xaxes(title_text="Position", row=1, col=1)
fig.update_xaxes(title_text="Position", row=1, col=2)
fig3.update_layout(title={
        'text': 'Interference mean profile',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24}},
        font_family="Raleway",
        height=600,
        width=1000)
fig3.show()
fig3.write_html("Interference mean profile.html")
