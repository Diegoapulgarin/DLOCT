import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
t = np.linspace(1,1000,10000)
f = 1 * np.cos(2*np.pi*10*t)
plt.plot(f)
plt.show(block=True)
print(len(t))

# fig = px.line(f)
# fig.show()