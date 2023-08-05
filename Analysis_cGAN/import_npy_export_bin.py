
#%%
import numpy as np 
import matplotlib.pyplot as plt
#%%
load_path = 'D:\DLOCT\TomogramsDataAcquisition\ExperimentalTomogram'
npy_file = 'ExperimentalROI_corrected5_DL_resampled_of'
binfile = np.load(load_path+'\\'+ npy_file+'.npy')
imag = np.imag(binfile)
real = np.real(binfile)
plot = 10*np.log(abs(real[256,:,:]+1j*imag[256,:,:])**2)
plt.imshow(plot)

#%%
file_path = load_path + '\\' + npy_file + '_imag.bin'
imag.astype(np.float64).tofile(file_path)
# fid = open(load_path + '\\' + npy_file + '_imag.bin', "wb")
# # imag_f_order = np.asfortranarray(imag)
# imag_c_contiguous = np.ascontiguousarray(imag.astype(np.float64))
# fid.write(imag_c_contiguous.tobytes())
# fid.close()

print('imag saved')
file_path = load_path + '\\' + npy_file + '_real.bin'
real.astype(np.float64).tofile(file_path)
# fid = open(load_path + '\\' + npy_file + '_real.bin', "wb")
# # real_f_order = np.asfortranarray(real)
# real_c_contiguous = np.ascontiguousarray(real.astype(np.float64))
# fid.write(real_c_contiguous.tobytes())
# fid.close()
print('real saved')
#%%
import scipy.io

# Assuming load_path, npy_file, and imag are defined before this line
file_path = load_path + '\\' + npy_file + '_imag.mat'

# Save the ndarray as float64 (double) data type in a .mat file
scipy.io.savemat(file_path, {'imag_data': imag.astype(np.float64)})


# Assuming load_path, npy_file, and imag are defined before this line
file_path = load_path + '\\' + npy_file + '_real.mat'

# Save the ndarray as float64 (double) data type in a .mat file
scipy.io.savemat(file_path, {'real_data': real.astype(np.float64)})

