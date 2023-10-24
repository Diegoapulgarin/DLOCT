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

#%%
thisfringes = 2
fringesTest = fringes[thisfringes,:,:,:]
real_fringe = np.abs(fringesTest) * np.cos(np.angle(fringesTest))
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 20  # Puedes ajustar este valor según tus necesidades
smoothed_envelope = moving_average(envelope, window_size)

second_derivative_smoothed = np.diff(smoothed_envelope, n=2)
zero_crossings_smoothed = np.where(np.diff(np.sign(second_derivative_smoothed)))[0]

min_distance = 10  # Ajusta este valor según tus necesidades
consolidated_crossings = [zero_crossings_smoothed[0]]

for i in range(1, len(zero_crossings_smoothed)):
    if zero_crossings_smoothed[i] - consolidated_crossings[-1] > min_distance:
        consolidated_crossings.append(zero_crossings_smoothed[i])

consolidated_crossings = np.array(consolidated_crossings)

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

# Transformada de Hilbert para obtener la señal analítica
analytic_signal = hilbert(real_fringe[:,1,1])

# Calculamos la fase de la señal analítica
phase_real_fringes = np.angle(analytic_signal)

# Inicializamos nuestra fase acumulada con la fase de real_fringes
accumulated_phase = phase_real_fringes.copy()

for i in range(1, len(segmented_fringes_without_padding)):
    # Calculamos la fase del fragmento actual
    current_phase = np.angle(segmented_fringes_without_padding[i])
    
    # Actualizamos la fase acumulada a partir de ese fragmento
    accumulated_phase[i:i+len(current_phase)] += current_phase
    

# Finalmente, normalizamos la fase acumulada para que esté entre -pi y pi
accumulated_phase = np.angle(np.exp(1j * accumulated_phase))

complex_signal = real_fringe[:,1,1] * np.exp(1j * accumulated_phase)


#%% comprobación fft
fft_fringetest = np.fft.fftshift(np.fft.fft(fringesTest[:,1,1]))
fft_complexsignal = np.fft.fftshift(np.fft.fft(complex_signal))
fft_realfringe = np.fft.fftshift(np.fft.fft(real_fringe[:,1,1]))


# Crear un subplot con 2 filas y 1 columna
fig = make_subplots(rows=3, cols=1)

# Agregar la primera traza al primer subplot
fig.add_trace(go.Scatter(y=abs(fft_fringetest), mode='lines', name='fft original complex_signal'), row=1, col=1)

# Agregar la segunda traza al segundo subplot
fig.add_trace(go.Scatter(y=abs(fft_complexsignal), mode='lines', name='fft estimated complex_signal'), row=2, col=1)

fig.add_trace(go.Scatter(y=abs(fft_realfringe), mode='lines', name='fft real signal'), row=3, col=1)

# Mostrar la figura
fig.show()


fig=go.Figure()
fig.add_trace(go.Scatter(y=(np.imag(fringesTest[:,1,1])), mode='lines', name='imag part original'))
fig.add_trace(go.Scatter(y=(np.imag(complex_signal)), mode='lines', name='imag estimated'))
fig.show()



