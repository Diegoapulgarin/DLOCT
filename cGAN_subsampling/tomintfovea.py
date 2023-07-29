#%%
import numpy as np
import plotly.express as px
#%%
path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected'
tomint = np.fromfile(path+'\[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomIntDown_z=(586)_x=(448)_y=(960).bin','single')
tom = np.reshape(tomint,(586,896,960),order='F')
#%%
pos = 536
plot = 10*np.log10(tom[:,:,536])
fig = px.imshow(plot,color_continuous_scale='gray',zmin=70,zmax=150)
fig.show()
# fig.write_html('bscan.html')
# fig.write_image('bscan.svg')
#%%
value = tom[:,:,536,0]