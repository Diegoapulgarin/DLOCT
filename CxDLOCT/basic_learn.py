#%%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert
import plotly.express as px
#%%
fs = 300
f = 10
w = 2*np.pi*f
t = np.arange(0,1,1/fs)
A = 1
phase = 0
srv = A*np.cos(w*t+phase)
scv = A*np.exp(1j*(w*t+phase))
real_signal = np.real(scv) 
analytic_signal = hilbert(real_signal)
imaginary_signal = analytic_signal.imag

recscv = real_signal + 1j*imaginary_signal
fftsrv = abs(np.fft.fftshift(np.fft.fft(srv)))
fftscv = abs(np.fft.fftshift(np.fft.fft(scv)))
fftrec = abs(np.fft.fftshift(np.fft.fft(recscv)))
srvfreq = np.fft.fftshift(np.fft.fftfreq((len(fftscv)),1/fs))

fig = make_subplots(rows=4,cols=1)
fig.add_trace(go.Line(y=srv,x=t,name='Cosine'),row=1,col=1)
fig.add_trace(go.Line(y=np.real(scv),x=t,line=dict(dash='dash'),name='real part'),row=1,col=1)
fig.add_trace(go.Line(y=np.imag(scv),x=t,line=dict(dash='dot'),name='imag part'),row=1,col=1)
fig.add_trace(go.Line(y=imaginary_signal,x=t,line=dict(dash='dash'),name='recovered imaginarie signal'),row=1,col=1)
# show fft
fig.add_trace(go.Line(y=fftsrv,x=srvfreq,name='FFT cosine'),row=2,col=1)
fig.add_trace(go.Line(y=fftrec,x=srvfreq,name='FFT recovered signal'),row=3,col=1)
fig.add_trace(go.Line(y=fftscv,x=srvfreq,name='FFT exponential'),row=4,col=1)
fig.show()
# fig.write_html('Basic_complex_problem.html')

#%%

intensity = np.abs(scv)*np.cos(np.angle(scv))
fft_intensity = abs(np.fft.fftshift(np.fft.fft(intensity)))
fig=px.line(y = fft_intensity,x=srvfreq)
fig.show()

#%%
analytic_signal2 = hilbert(intensity)
recscv2 = intensity + 1j*analytic_signal2.imag
fft_intensity2 = abs(np.fft.fftshift(np.fft.fft(recscv2)))
fig=px.line(y = fft_intensity2, x = srvfreq)
fig.show()

#%%
amp = abs(recscv2)
phase = np.angle(recscv2)
crec = amp*np.cos(phase)
fig = make_subplots(rows=2,cols=1)
fig.add_trace(go.Line(y=amp,x=t,name='Amplitude'),row=1,col=1)
fig.add_trace(go.Line(y=phase,x=t,name='phase'),row=1,col=1)
fig.add_trace(go.Line(y=crec,x=t,name = 'cosine'),row=2,col=1)
fig.show()
