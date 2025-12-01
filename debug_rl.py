import numpy as np
import matplotlib.pyplot as plt
from mtf_aware_deblurring.reconstruction.richardson_lucy import richardson_lucy
from mtf_aware_deblurring.optics import fft_convolve2d, motion_psf_from_code, kernel2d_from_psf1d
from mtf_aware_deblurring.metrics import psnr

def test_rl():
    # Create synthetic scene
    scene = np.zeros((256, 256), dtype=np.float32)
    scene[64:192, 64:192] = 1.0
    
    # Create kernel
    code = np.ones(31) # Box code
    psf = motion_psf_from_code(code, length_px=15.0)
    kernel = kernel2d_from_psf1d(psf)
    
    # Blur
    blurred = fft_convolve2d(scene, kernel)
    
    # Run RL
    recon = richardson_lucy(blurred, kernel, iterations=30)
    
    p = psnr(scene, recon)
    print(f"RL PSNR: {p:.2f} dB")
    
    # Save debug images
    plt.imsave("debug_scene.png", scene, cmap="gray")
    plt.imsave("debug_blurred.png", blurred, cmap="gray")
    plt.imsave("debug_recon.png", recon, cmap="gray")

if __name__ == "__main__":
    test_rl()
