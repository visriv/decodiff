import math
import numpy as np

def get_spatial_spectrum(matrix):
    H, W = matrix.shape[0], matrix.shape[1]
    # Round up the size along each axis to an even number
    n_H = int(math.ceil(H / 2.) * 2)
    n_W = int(math.ceil(W / 2.) * 2)
    
    # Compute the 2D Fourier transform for gt and pred using rfft along both axes
    fft_gt_x = np.fft.rfft(matrix, n=n_W, axis=1)
    fft_gt_y = np.fft.rfft(matrix, n=n_H, axis=0)


    
    # Compute power spectrum (multiply by complex conjugate)
    energy_gt_x = fft_gt_x.real ** 2 + fft_gt_x.imag ** 2
    energy_gt_y = fft_gt_y.real ** 2 + fft_gt_y.imag ** 2



    # Average over appropriate axes
    energy_gt_x = energy_gt_x.sum(axis=0) / fft_gt_x.shape[0]
    energy_gt_y = energy_gt_y.sum(axis=1) / fft_gt_y.shape[1]


    # Combine energies for a single spectrum
    energy_gt = 0.5 * (energy_gt_x + energy_gt_y)


    # Generate wavenumber axis (only for positive frequencies)
    wavenumber_x = np.fft.rfftfreq(n_W)
    wavenumber_y = np.fft.rfftfreq(n_H)

    # Since energy_gt and energy_pred are averaged over x and y axes, 
    # we use the wavenumber from one of the axes (they represent equivalent ranges)
    wavenumber = wavenumber_x  # or wavenumber_y, since the final energy spectrum is averaged

    # Sort wavenumber and energy values in descending order
    sorted_indices = np.argsort(wavenumber)[::-1]
    wavenumber_sorted = wavenumber[sorted_indices]
    energy_gt_sorted = energy_gt[sorted_indices]

    return energy_gt_sorted, wavenumber_sorted