import numpy as np

def LowNA_3D(amp, z, x, y, k_vect, k, xi_x, xi_y, alpha, z_fp, z_ref, max_batch_size=None):
    # Beam waist diameter
    beam_waist_diam = 2 * alpha / k
    # Raylight range
    z_r = 2 * alpha ** 2 / k

    # Remove all points beyond n times the beam position; their contribution
    # is not worth the calculation
    null_points = np.sqrt(x ** 2 + y ** 2) > 20 * beam_waist_diam * np.sqrt(1 + (z / z_r) ** 2)
    amp = np.delete(amp, null_points)
    z = np.delete(z, null_points)
    x = np.delete(x, null_points)
    y = np.delete(y, null_points)

    # Number of points
    n_points = z.shape[2]
    
    # If not input batchSize calculate contribution from all points at once
    if max_batch_size is None:
        max_batch_size = n_points
    
    # Batch size
    batch_size = min(max_batch_size, n_points)
    
    # Number of batches of points
    n_batches = np.ceil(n_points / batch_size).astype(int)
    
    # Initialize output
    fringes = np.zeros((k_vect.shape[0], 1), dtype=k_vect.dtype)
    
    # Iterate batches
    for j in range(n_batches):
        # Calculate the contribution from this batch of points
        this_batch = np.unique(np.arange(batch_size) + (j * batch_size), return_index=True)[1]
        this_batch = np.take(this_batch, np.where(this_batch < n_points))
        
        # In this case we use 2*kVect - xi_x^2/(4*k) where kVect is a vector BUT
        # k is an scalar, yielding the low-NA model
        this_fringes = 1 / (8 * np.pi ** 2) / \
          ((alpha / k) ** 2 + (1j * (z[:, :, this_batch] - z_fp) / k)) * \
          np.exp(2j * (z[:, :, this_batch] - z_ref) * k_vect) * \
          np.sum(np.exp(-1j * (xi_x * x[:, :, this_batch])) * \
          np.exp(-1j * (z[:, :, this_batch] - z_fp) * xi_x ** 2 / k / 4) * \
          np.exp(- (xi_x * alpha / k / 2) ** 2), axis=1) * \
          np.sum(np.exp(-1j * (xi_y * y[:, :, this_batch])) * \
          np.exp(-1j * (z[:, :, this_batch] - z_fp) * xi_y ** 2 / k / 4) * \
          np.exp(- (xi_y * alpha / k / 2) ** 2), axis=3)
        
        # sum the contribution of all scatterers, considering its individual
        # amplitudes
        fringes = fringes + np.sum(amp[:, :, this_batch] * this_fringes, axis=2)
    
    return fringes
