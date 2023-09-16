#%%
import numpy as np 
import plotly.graph_objects as go
path = r'C:\Users\USER\Documents\GitHub\Fovea'
file = '\\cGAN_1_metrics_log.npy'
metrics = np.load(path+file)
#%%
'''
    metrics in file
    0 ssims
    1 ssims_std
    2 ssims_uncertainty
    3 phasemetric
    4 phasemetricCorrected
    5 mse
    6 mse_std
    7 mse_uncertainty

'''
ssim_metric = metrics[:,0]
ssim_std = metrics[:,1]
ssim_uncertainty = metrics[:,2]
figure = go.Figure()
figure.add_trace(go.Line(y=ssim_metric))
# figure.add_trace(go.Line(y=ssim_metric + ssim_std))
# figure.add_trace(go.Line(y=ssim_metric - ssim_std))
figure.add_trace(go.Line(y=ssim_metric +ssim_uncertainty))
figure.show()
