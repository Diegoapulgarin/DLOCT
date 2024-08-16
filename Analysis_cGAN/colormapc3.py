#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
pathCmap = r'C:\Users\USER\Documents\GitHub\DLOCT'
file = 'c3_colormap.csv'
c3 = pd.read_csv(os.path.join(pathCmap,file),sep=' ',header=None)
custom_cmap = mcolors.ListedColormap(np.array(c3))

# Para asegurarse de que el colormap cubre toda la gama, puede utilizar BoundaryNorm
# norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 1, len(c3) + 1), ncolors=len(c3))

# Ejemplo de uso del colormap para graficar una imagen
data = np.random.rand(10, 10)  # Data aleatoria para ilustrar
plt.imshow(data, cmap=custom_cmap)
plt.colorbar()
plt.show()