#%%
import numpy as np
import plotly.graph_objects as go

# Generate the x-axis
x = np.linspace(-100, 100, 1000)

# Generate the signal: a combination of four cosine functions and an offset to simulate the DC component
signal = (np.cos(2 * np.pi * 0.1 * x) + 0.5*np.cos(2 * np.pi * 0.15 * x) + 
          np.cos(2 * np.pi * 1.5 * x) + np.cos(2 * np.pi * 2 * x)) + 1.5  # Offset to introduce DC component

# Apply FFT to the signal
fft_signal = np.fft.fftshift(np.abs(np.fft.fft(signal)))

# Frequency axis for the FFT result
frequencies = np.fft.fftshift(np.fft.fftfreq(len(x), (x[1] - x[0])))

# Plot the signal
fig = go.Figure()
fig.add_trace(go.Scatter(x=frequencies, y=fft_signal, mode='lines', line=dict(color='purple')))

# Update layout
fig.update_layout(
    title='Simulated Signal FFT',
    xaxis_title='Frequency',
    yaxis_title='Amplitude',
    showlegend=False
)

# Display the plot
fig.show()

#%%
import numpy as np
import plotly.graph_objects as go

# Generate x-axis (normalized delay τ/τ_c)
tau = np.linspace(-5, 5, 1000)

# Generate g^(1) functions: coherent, Lorentzian, and Gaussian
g_coherent = np.ones_like(tau)  # Coherent state (ideal laser)
g_lorentzian = 1 / (1 + tau**2)  # Lorentzian chaotic light
g_gaussian = np.exp(-tau**2)  # Gaussian chaotic light

# Create the plot
fig = go.Figure()

# Add traces for each g^(1) function
fig.add_trace(go.Scatter(x=tau, y=g_coherent, mode='lines', name='Coherente (láser ideal)', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=tau, y=g_lorentzian, mode='lines', name='Caótica Lorentziana', line=dict(color='red')))
fig.add_trace(go.Scatter(x=tau, y=g_gaussian, mode='lines', name='Caótica Gaussiana', line=dict(color='green')))

# Update layout
fig.update_layout(
    # title='Primer Grado de Coherencia |g<sup>(1)</sup>|',
    xaxis_title='τ / τ<sub>c</sub>',
    yaxis_title='|g<sup>(1)</sup>|',
    showlegend=True
)

# Display the plot
fig.show()

