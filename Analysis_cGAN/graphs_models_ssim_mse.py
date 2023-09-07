#%%
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp

path = r'C:\Users\USER\Documents\cGAN_1'
log_metrics = np.load(path+'\\'+'metrics_log.npy')
ssim = log_metrics[0:81,0]
mse = log_metrics[0:81,3]
phaseMetric = log_metrics[:,1]
phasemetricCorrected = log_metrics[:,2]
fig = sp.make_subplots(rows=2, cols=1)

fig.add_trace(go.Line(y=ssim, name='SSIM index'), row=1, col=1)
fig.add_trace(go.Line(y=mse, name='MSE index'), row=2, col=1)

# Update x-axis and y-axis labels
fig.update_xaxes(title_text="Epochs", row=1, col=1)
fig.update_yaxes(title_text="SSIM", row=1, col=1) # SSIM is dimensionless

fig.update_xaxes(title_text="Epochs", row=2, col=1)
fig.update_yaxes(title_text="MSE error", row=2, col=1) # MSE is dimensionless unless it's normalized
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=12,
        color="Black"
    )
)
fig.show()
fig.write_html(r'C:\Users\USER\Documents\GitHub\DLOCT\Analysis_cGAN\Fast_evaluatecGAN_1.html')