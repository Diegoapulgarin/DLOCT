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


thisfringes = 2
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

# Asumimos que 'peaks' y 'properties' han sido definidos previamente utilizando find_peaks

# Usamos 'left_ips' y 'right_ips' para obtener los puntos de inicio y fin de cada pico
left_bases = properties["left_bases"]
right_bases = properties["right_bases"]

# Inicializamos la lista para guardar los segmentos expandidos
expanded_segments = []

# La primera expansión comienza desde el primer punto de la señal
start_expansion = 0

for i in range(len(peaks)):
    # Definimos el punto de inicio de este segmento como el máximo entre el inicio del pico y la expansión actual
    start = max(left_bases[i], start_expansion)
    # Definimos el punto de fin de este segmento como el mínimo entre el final del pico y el inicio del próximo pico
    end = min(right_bases[i], left_bases[i+1] if i+1 < len(peaks) else len(accumulated_phase))
    
    # Añadimos el segmento expandido a la lista
    expanded_segments.append(accumulated_phase[start:end])

    # El próximo inicio de expansión será el final de este segmento
    start_expansion = end

# Verificamos si la longitud total de los segmentos expandidos es igual a la longitud de la señal de fase acumulada
assert sum(len(segment) for segment in expanded_segments) == len(accumulated_phase), "La longitud de los segmentos no concuerda con la señal de fase acumulada."

# Regresamos la cantidad de segmentos y una muestra de los segmentos
len(expanded_segments), expanded_segments[:2]  # Mostramos solo los primeros dos para verificar


# fig=go.Figure()
# fig.add_trace(go.Scatter(y=(np.imag(fringesTest[:,1,1])), mode='lines', name='imag part original'))
# fig.add_trace(go.Scatter(y=(np.imag(complex_signal)), mode='lines', name='imag estimated'))
# fig.show()



