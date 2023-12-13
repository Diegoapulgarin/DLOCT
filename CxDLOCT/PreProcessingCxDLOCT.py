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
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from PyEMD import EMD

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def segment_signal_without_padding(signal, crossings):
    fragments = []
    for i in range(0, len(crossings) - 1, 2):  # Tomamos los puntos de inflexión en pares
        start, end = crossings[i], crossings[i + 1]
        
        # Extrae el segmento de interés
        fragment = signal[start:end]
        
        # Añade el fragmento a la lista de fragmentos
        fragments.append(fragment)
        
    return fragments  # Nota: ahora la lista no es convertida a un array porque los fragmentos tienen diferentes longitudes

#%% reading  experimental tomograms

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



#%% calculate and plot de envelope


thisfringes = 6
fringesTest = fringes[thisfringes,:,:,:]
real_fringe = np.abs(fringesTest) * np.cos(np.angle(fringesTest))
emd = EMD()
IMFs = emd(real_fringe[:,1,1])

# Elije una IMF específica, por ejemplo la primera:
imf_selected = IMFs[0]
combined_IMF = np.sum(IMFs[:6], axis=0)
# Obtener la señal analítica
analytic_signal = hilbert(combined_IMF)

# Calcular la envolvente
envelope = np.abs(analytic_signal)

fig = go.Figure()
fig.add_trace(go.Scatter(y=(envelope), mode='lines', name='+ envelope',
                         line=dict(dash='dash')))
fig.add_trace(go.Scatter(y=(real_fringe[:,1,1]), mode='lines', name='fringe'))
fig.add_trace(go.Scatter(y=(-envelope), mode='lines', name='- envelope',
                         line=dict(dash='dash')))
fig.show()




#%% smooth the envelope 
thisfringes = 6
fringesTest = fringes[thisfringes,:,:,:]
real_fringe = np.abs(fringesTest) * np.cos(np.angle(fringesTest))


window_size = 3  # Puedes ajustar este valor según tus necesidades
smoothed_envelope = moving_average(envelope, window_size)
px.line(smoothed_envelope)
#%%

second_derivative_smoothed = np.diff(smoothed_envelope, n=2)
zero_crossings_smoothed = np.where(np.diff(np.sign(second_derivative_smoothed)))[0]

min_distance = 10  # Ajusta este valor según tus necesidades
consolidated_crossings = [zero_crossings_smoothed[0]]

for i in range(1, len(zero_crossings_smoothed)):
    if zero_crossings_smoothed[i] - consolidated_crossings[-1] > min_distance:
        consolidated_crossings.append(zero_crossings_smoothed[i])

consolidated_crossings = np.array(consolidated_crossings)

segmented_fringes_without_padding = segment_signal_without_padding(real_fringe[:,1,1], consolidated_crossings)
px.line(second_derivative_smoothed)
#%%
analytic_signal = hilbert(real_fringe[:,1,1])
phase_real_fringes = np.angle(analytic_signal)
accumulated_phase = phase_real_fringes.copy()
ubication = len(segmented_fringes_without_padding[0])+1
for i in range(1, len(segmented_fringes_without_padding)):
    # Calculamos la fase del fragmento actual
    print(ubication)
    current_phase = np.angle(segmented_fringes_without_padding[i])
    
    # Actualizamos la fase acumulada a partir de ese fragmento
    accumulated_phase[ubication:ubication+len(current_phase)] += current_phase
    ubication +=(len(current_phase)+1)

# Finalmente, normalizamos la fase acumulada para que esté entre -pi y pi
accumulated_phase = np.angle(np.exp(1j * accumulated_phase))

complex_signal = real_fringe[:,1,1] * np.exp(1j * accumulated_phase)


fft_fringetest = np.fft.fftshift(np.fft.fft(fringesTest[:,1,1]))
fft_complexsignal = np.fft.fftshift(np.fft.fft(complex_signal))
fft_realfringe = np.fft.fftshift(np.fft.fft(real_fringe[:,1,1]))



fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(y=abs(fft_fringetest), mode='lines', name='fft original complex_signal'), row=2, col=1)
fig.add_trace(go.Scatter(y=abs(fft_complexsignal), mode='lines', name='fft estimated complex_signal'), row=2, col=1)
fig.add_trace(go.Scatter(y=abs(fft_realfringe), mode='lines', name='fft real signal'), row=1, col=1)
fig.show()

#%%
import numpy as np
from scipy.signal import find_peaks

# Supongamos que 'smoothed_envelope' y 'accumulated_phase' están pre-cargados con los datos relevantes
# smoothed_envelope = ...
# accumulated_phase = ...

# Encuentra picos y valles con prominencia y anchura adecuadas
peaks, properties = find_peaks(smoothed_envelope, prominence=1, width=5)
valleys, _ = find_peaks(-smoothed_envelope, prominence=1, width=5)

# Utiliza 'left_ips' y 'right_ips' para obtener los puntos de inicio y fin completos de cada pico
segments = []
for left, right in zip(properties["left_ips"], properties["right_ips"]):
    segment = accumulated_phase[int(left):int(right)]
    segments.append(segment)

# Visualización con Matplotlib
import matplotlib.pyplot as plt
plt.plot(smoothed_envelope)
plt.plot(peaks, smoothed_envelope[peaks], "x")
plt.vlines(x=peaks, ymin=smoothed_envelope[peaks] - properties["prominences"], ymax=smoothed_envelope[peaks], color="C1")
plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color="C1")
plt.show()
#%%
# Aquí tu código ajustado para dividir los espacios entre segmentos equitativamente.

# Inicializamos la lista para guardar los índices de inicio y fin de cada segmento
segment_indices = []

# El primer segmento comienza en el primer punto de la señal
start_index = 0

for i in range(len(peaks) - 1):
    # El final de un segmento es el punto medio entre el pico actual y el siguiente
    end_index = (peaks[i] + peaks[i+1]) // 2
    segment_indices.append((start_index, end_index))
    
    # El inicio del siguiente segmento es el final del segmento actual
    start_index = end_index

# Añadir el último segmento que va desde el último punto medio hasta el final de la señal
segment_indices.append((start_index, len(accumulated_phase)))

# Ahora extraemos los segmentos de la señal de fase acumulada basándonos en los índices
expanded_segments = [accumulated_phase[start:end] for start, end in segment_indices]

# Calculamos la longitud total de los segmentos expandidos para verificar
total_length_segments = sum(len(segment) for segment in expanded_segments)

# Comparamos la longitud total de los segmentos con la longitud de la señal de fase acumulada
if total_length_segments != len(accumulated_phase):
    print(f"La longitud total de los segmentos ({total_length_segments}) no coincide con la longitud de la señal acumulada ({len(accumulated_phase)}).")
else:
    print("La longitud de los segmentos coincide con la señal de fase acumulada.")
#%%

# Inicializamos la fase acumulada como un número complejo
accumulated_phase_complex = np.zeros_like(accumulated_phase, dtype=complex)

for segment in expanded_segments:
    # Calcula la transformada de Hilbert del segmento
    analytic_signal = hilbert(segment)
    
    # Calcula la fase del segmento
    phase_segment = np.angle(analytic_signal)
    
    # Convierte la fase a su representación compleja
    complex_representation = np.exp(1j * phase_segment)
    
    # Suma fasorial: sumar la representación compleja con la fase acumulada
    accumulated_phase_complex[:len(segment)] += complex_representation
    
    # Para la parte de la señal que no está cubierta por el segmento actual, 
    # mantenemos la última fase acumulada
    if len(segment) < len(accumulated_phase):
        accumulated_phase_complex[len(segment):] = accumulated_phase_complex[len(segment) - 1]

# Convertir la representación compleja acumulada de nuevo a fase
accumulated_phase_total = np.angle(accumulated_phase_complex)


#%%
thisfringes = 4
fringesTest = fringes[thisfringes,:,:,:]
real_fringe = np.abs(fringesTest) * np.cos(np.angle(fringesTest))
# Recomponer la señal compleja
recomposed_complex_signal = np.abs(real_fringe[:,1,1]) * np.exp(1j * accumulated_phase_total)
fft_fringetest = np.fft.fftshift(np.fft.fft(fringesTest[:,1,1]))
fft_complexsignal = np.fft.fftshift(np.fft.fft(recomposed_complex_signal))
fft_realfringe = np.fft.fftshift(np.fft.fft(real_fringe[:,1,1]))



fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(y=abs(fft_fringetest), mode='lines', name='fft original complex_signal'), row=2, col=1)
fig.add_trace(go.Scatter(y=abs(fft_complexsignal), mode='lines', name='fft estimated complex_signal'), row=2, col=1)
fig.add_trace(go.Scatter(y=abs(fft_realfringe), mode='lines', name='fft real signal'), row=1, col=1)
fig.show()
# Ahora 'accumulated_phase_total' contiene la fase acumulativa total


# fig=go.Figure()
# fig.add_trace(go.Scatter(y=(np.imag(fringesTest[:,1,1])), mode='lines', name='imag part original'))
# fig.add_trace(go.Scatter(y=(np.imag(complex_signal)), mode='lines', name='imag estimated'))
# fig.show()



