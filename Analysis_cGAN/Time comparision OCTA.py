#%%
import numpy as np
import os
import matplotlib.pyplot as plt
#%%

path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
fnameTom = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomoriginalFlat_z=400_x=896_y=960_pol=2' # fovea
tomShape = [(400,896,960,2)]# porcine cornea
fname = os.path.join(path, fnameTom)
# Names of all real and imag .bin files
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]
tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using