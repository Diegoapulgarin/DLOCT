def ForwardModel_PointScatterers_HighNA(amp, z, x, k_vect, xi, alpha, z_fp, z_ref, max_batch_size=None):
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
        
        # In this case we use sqrt((2*k)^2 - xi^2) where k is a vector
        this_fringes = (1 / (8 * np.pi) / \
          ((alpha / k_vect) ** 2 + (1j * (z[:, :, this_batch] - z_ref) / k_vect)) * \
          np.sum(np.exp(-1j * xi * x[:, :, this_batch] + \
          (1j * (z[:, :, this_batch] - z_ref) * np.sqrt((2 * k_vect) ** 2 - xi ** 2)) + \
          (- (xi * alpha / k_vect / 2) ** 2)), 1))
        
        # sum the contribution of this batch of scatterers, considering its
        # individual amplitudes, and sum to the overall contribution
        fringes = fringes + np.sum(amp[:, :, this_batch] * this_fringes, axis=2)
    
    return fringes