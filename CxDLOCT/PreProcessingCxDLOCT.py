#%%
''' 
this script is about pre-processing phase 
of DLOCT for complex conjugate removal

'''

import numpy as np
import scipy.io as sio
import plotly.express as px
import os
import plotly.graph_objs as go
import plotly.subplots as sp
from scipy.signal import hilbert

#%%

path = r'C:\Users\USER\Documents\GitHub\Simulated_Data_Complex'
os.chdir(path)
fringes = []
for filename in os.listdir(os.getcwd()):
   print(path+'/'+filename)
   mat_contents = sio.loadmat(path+'/'+filename)
   fringes1 = mat_contents['fringes1']
   divisions = int(fringes1.shape[2]/16)
   n = 0 
   for i in range(divisions):
       fringes_slice = fringes1[:, :, n:n+16]
       n = n + 16
       fringes.append(fringes_slice)
   print(filename)
fringes = np.array(fringes)
del fringes1, fringes_slice
#%%
thisfringes = 0
fringesTest = fringes[thisfringes,:,:,:]
real_fringe = np.abs(fringesTest) * np.cos(np.angle(fringesTest))


analytic_signal = hilbert(real_fringe)

envelope = np.abs(analytic_signal)
#%%

fig = go.Figure()
fig.add_trace(go.Scatter(y=(envelope[:,1,1]), mode='lines', name='+ envelope',
                         line=dict(dash='dash')))
fig.add_trace(go.Scatter(y=(real_fringe[:,1,1]), mode='lines', name='fringe'))
fig.add_trace(go.Scatter(y=(-envelope[:,1,1]), mode='lines', name='- envelope',
                         line=dict(dash='dash')))
fig.show()

#%%

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 20  # Puedes ajustar este valor según tus necesidades
smoothed_envelope = moving_average(envelope[:,1,1], window_size)

second_derivative_smoothed = np.diff(smoothed_envelope, n=2)
zero_crossings_smoothed = np.where(np.diff(np.sign(second_derivative_smoothed)))[0]

min_distance = 10  # Ajusta este valor según tus necesidades
consolidated_crossings = [zero_crossings_smoothed[0]]

for i in range(1, len(zero_crossings_smoothed)):
    if zero_crossings_smoothed[i] - consolidated_crossings[-1] > min_distance:
        consolidated_crossings.append(zero_crossings_smoothed[i])

consolidated_crossings = np.array(consolidated_crossings)

def segment_signal(signal, crossings):
    fragments = []
    for i in range(0, len(crossings) - 1, 2):  # Tomamos los puntos de inflexión en pares
        fragment = np.zeros_like(signal)
        start, end = crossings[i], crossings[i + 1]
        
        # Copia la señal del segmento en el fragmento
        fragment[start:end] = signal[start:end]
        
        # Añade el fragmento a la lista de fragmentos
        fragments.append(fragment)
        
    return np.array(fragments)

# Segmentamos la señal
segmented_fringes = segment_signal(real_fringe[:,1,1], consolidated_crossings)

# Verificamos las dimensiones de los segmentos obtenidos
print(segmented_fringes.shape)
#%%
def segment_signal_without_padding(signal, crossings):
    fragments = []
    for i in range(0, len(crossings) - 1, 2):  # Tomamos los puntos de inflexión en pares
        start, end = crossings[i], crossings[i + 1]
        
        # Extrae el segmento de interés
        fragment = signal[start:end]
        
        # Añade el fragmento a la lista de fragmentos
        fragments.append(fragment)
        
    return fragments  # Nota: ahora la lista no es convertida a un array porque los fragmentos tienen diferentes longitudes

# Segmentamos la señal sin zero padding
segmented_fringes_without_padding = segment_signal_without_padding(real_fringe[:,1,1], consolidated_crossings)

# Verificamos la cantidad de segmentos obtenidos
print(len(segmented_fringes_without_padding))


#%%

import numpy as np
from scipy.signal import hilbert

# 1. Calcular la Transformada de Hilbert de cada fragmento
analytic_signals = [hilbert(fragment) for fragment in segmented_fringes_without_padding]

# 2. Extraer la fase de cada fragmento
phases = [np.angle(analytic_signal) for analytic_signal in analytic_signals]

# 3. Acumulación de la fase
accumulated_phase = np.zeros_like(real_fringe[:,1,1])
start_idx = 0
for i, phase in enumerate(phases):
    end_idx = start_idx + len(phase)
    if i == 0:
        accumulated_phase[start_idx:end_idx] = phase
    else:
        accumulated_phase[start_idx:end_idx] = phase + accumulated_phase[start_idx - 1]
    start_idx = end_idx

# 4. Construir la señal compleja acumulada
amplitude = np.abs(real_fringe[:,1,1])
complex_accumulated_signal = amplitude * np.exp(1j * accumulated_phase)

# Verificar las dimensiones
print(complex_accumulated_signal.shape)


#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Crear un subplot con 2 filas y 1 columna
fig = make_subplots(rows=2, cols=1)

# Agregar la primera traza al primer subplot
fig.add_trace(go.Scatter(y=segmented_fringes[1,:], mode='lines', name='Fragmento'), row=1, col=1)

# Agregar la segunda traza al segundo subplot
fig.add_trace(go.Scatter(y=real_fringe[:,1,1], mode='lines', name='Fringe Completa'), row=2, col=1)

# Mostrar la figura
fig.show()

