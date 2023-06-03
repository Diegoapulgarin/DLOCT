import numpy as np
def LowNA_3D(amp, z, x, y, k_vect, k, xi_x, xi_y, alpha, z_fp, z_ref, max_batch_size):
    # total number of scatterers
    n_points = len(z)
    # create a list of all scatterers
    scatterers_idx = np.arange(n_points)
    # initialize fringes
    fringes = np.zeros(k_vect.shape, dtype=np.complex128)
    # iterate in batches
    for i in range(0, n_points, max_batch_size):
        this_batch = scatterers_idx[i:i + max_batch_size]
        this_batch = np.take(this_batch, np.where(this_batch < n_points))
        # In this case we use 2*kVect - xi_x^2/(4*k) where kVect is a vector BUT
        # k is an scalar, yielding the low-NA model

        # Compute the first term of the product
        term1 = 1 / (8 * np.pi ** 2) / ((alpha / k) ** 2 + (1j * (z[this_batch] - z_fp) / k))
        # Reshape z and k_vect before computing the second term of the product
        z_res = z[this_batch].reshape(-1, 1) - z_ref
        k_vect_res = k_vect.reshape(1, -1)
        term2 = np.exp(2j * z_res * k_vect_res)

        # Compute the remaining terms as before
        term3 = np.sum(np.exp(-1j * (xi_x * x[this_batch])) * \
                    np.exp(-1j * (z[this_batch] - z_fp) * xi_x ** 2 / k / 4) * \
                    np.exp(- (xi_x * alpha / k / 2) ** 2))

        term4 = np.sum(np.exp(-1j * (xi_y * y[this_batch])) * \
                    np.exp(-1j * (z[this_batch] - z_fp) * xi_y ** 2 / k / 4) * \
                    np.exp(- (xi_y * alpha / k / 2) ** 2))

        # Now compute the product
        this_fringes = term1 * term2 * term3 * term4

        # sum the contribution of all scatterers, considering its individual
        # amplitudes
        fringes = fringes + np.sum(amp[this_batch] * this_fringes)
    return fringes

