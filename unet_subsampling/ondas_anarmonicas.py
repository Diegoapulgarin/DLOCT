#%%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
#%%

x = np.linspace(0,10,1000)
a1 = 3
a2 = 8
wave1 = a1*np.sin(np.pi*x)
wave2 = a2*np.sin((np.pi*x)/3)
wave3 = wave1 + wave2

fft = np.fft.fftshift(np.fft.fft(wave3))
xfreq = np.linspace(-100,100,1000)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x,
                            y=wave1,
                            mode='lines',
                            name='wave 1',
                                ))
fig.add_trace(go.Scatter(x=x,
                            y=wave2,
                            mode='lines',
                            name='wave 2',
                                ))
fig.add_trace(go.Scatter(x=x,
                            y=wave3,
                            mode='lines',
                            name='superposition',
                                ))
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=xfreq,
                            y=abs(fft),
                            mode='lines',
                            name='wave 3',
                                ))
fig2.show()
# %%
