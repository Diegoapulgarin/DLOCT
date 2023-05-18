#%%
from Deep_Utils import sliding_window, inv_sliding_window
import numpy as np
import plotly.express as px
#%%
tomData = np.ones((1,896,1024,2))
n = 128
window_size = (n,n)
step_size = (n,n)
slices = []
for i in range(len(tomData)):
    bslicei = sliding_window(tomData[i,:,:,0],window_size,step_size)
    bslicer = sliding_window(tomData[i,:,:,1],window_size,step_size)
    bslice = np.stack((bslicer,bslicei),axis=3)
    slices.append(bslice)
slices = np.array(slices)
slices = np.reshape(slices,(slices.shape[0]*slices.shape[1],slices.shape[2],slices.shape[3],slices.shape[4]))
#%%
original_size = (tomData.shape[1],tomData.shape[2])
original_planes = tomData.shape[0]
origslicesOver = np.reshape(slices,(original_planes,int(slices.shape[0]/original_planes),slices.shape[1],slices.shape[2],2),)
number_planes =  origslicesOver.shape[0]
tomDataOver = []
for b in range(number_planes):
    bslicei,_,_ = inv_sliding_window(origslicesOver[b,:,:,:,1],window_size,original_size,step_size)
    bslicer,_,_ = inv_sliding_window(origslicesOver[b,:,:,:,0],window_size,original_size,step_size)
    bslice = np.stack((bslicer,bslicei),axis=2)
    tomDataOver.append(bslice)
tomDataOver = np.array(tomDataOver)
fig = px.imshow(tomDataOver[0,:,:,0],color_continuous_scale='gray')
fig.show()