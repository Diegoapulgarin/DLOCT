#%%
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

# Parameters for the signal
frequency = 100  # Frequency of the original signal in Hz
sampling_rate_bad = 10  # Bad sampling rate (below Nyquist)

# Time vectors
t_cont = np.linspace(0, 0.1, 300)  # Continuous time vector for reference
t_bad = np.linspace(0, 0.1, sampling_rate_bad)  # Time vector for poorly-sampled signal

# Calculate the alias frequency correctly
k = round(frequency / sampling_rate_bad)  # Find the nearest integer k
alias_frequency_corrected = (frequency - k * sampling_rate_bad)  # Calculate alias frequency

# Signals
signal_cont = np.sin(2 * np.pi * frequency * t_cont)  # Original continuous signal
signal_bad = np.sin(2 * np.pi * frequency * t_bad)  # Poorly-sampled signal (scatter points)
signal_alias_corrected = np.sin(2 * np.pi * sampling_rate_bad * t_cont)  # Corrected alias signal passing through scatter points

# Create traces for Plotly
trace_cont = go.Scatter(x=t_cont, y=signal_cont, mode='lines', name='Señal de 100 Hz', line=dict(color='green'))
trace_bad = go.Scatter(x=t_bad, y=signal_bad, mode='markers', name='Muestreo Incorrecto', marker=dict(color='red'))
trace_alias_corrected = go.Scatter(x=t_cont, y=signal_alias_corrected, mode='lines', line=dict(dash='dash', color='blue'), name='Frecuencia Aliasing')

# Calculate the FFTs
fft_cont = np.fft.fft(signal_cont)
fft_bad = np.fft.fft(signal_bad, n=len(t_cont))  # Use the same length as t_cont for comparison
# Frequency vector for both FFTs
N_cont = len(t_cont)  # Number of points for continuous signal
freqs_cont = np.fft.fftfreq(N_cont, d=t_cont[1] - t_cont[0])

# Magnitude of FFTs
fft_cont_magnitude = np.abs(fft_cont)
fft_bad_magnitude = np.abs(fft_bad)

# Create subplots with 3 rows and 1 column
fig = make_subplots(
    rows=3, cols=1,
    vertical_spacing=0.15,
)

# Original signals in time domain
fig.add_trace(trace_cont, row=1, col=1)
fig.add_trace(trace_bad, row=1, col=1)
fig.add_trace(trace_alias_corrected, row=1, col=1)

# FFT plots
fig.add_trace(
    go.Scatter(x=freqs_cont, y=fft_cont_magnitude, mode='lines', name='FFT Muestreado a Nyquist', line=dict(color='orange')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=freqs_cont, y=fft_bad_magnitude, mode='lines', name='FFT Submuestreada', line=dict(color='purple')),
    row=3, col=1
)

# Update layout
fig.update_layout(
    height=700, width=1200,
    font=dict(size=20, family='Times New Roman'),
    legend=dict(
        font=dict(
            size=20,  # Aumentar el tamaño de la fuente de la leyenda
        ),
        orientation="h",  # Orientación horizontal
        yanchor="bottom",  # Anclaje de la leyenda
        y=-0.3,  # Ajustar la posición vertical (más abajo)
        xanchor="center",  # Anclaje horizontal centrado
        x=0.5,  # Posición horizontal centrada
        bgcolor="rgba(255, 255, 255, 0.8)",  # Fondo semitransparente para la leyenda (opcional)
        borderwidth=1  # Borde alrededor de la leyenda (opcional)
    ),
    margin=dict(
        t=50,  # Margen superior
        b=100,  # Margen inferior para espacio de la leyenda
    ),
)

# Update axes titles
fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
fig.update_yaxes(title_text="Amplitud", row=1, col=1)

fig.update_xaxes(title_text="Frecuencia (Hz)", row=2, col=1)
fig.update_yaxes(title_text="Magnitud FFT", row=2, col=1)

fig.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=1)
fig.update_yaxes(title_text="Magnitud FFT", row=3, col=1)

# Save the figure as an HTML file
path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Bibliografía complex conjugate mirror terms\imagenes_tesis'
fig.write_html(os.path.join(path, 'aliasing.html'))

fig.show()







