import imageio.v2 as imageio
import numpy as np
from pathlib import Path

def inspect():
    # Path to one of the noisy images from RL run
    # report/verification/noise1_rl_v2/0001x2/y_box_noisy.png
    # Note: run_forward_batches creates a subdir for each image if not collect-only?
    # Let's check the directory structure first.
    base = Path("report/verification/noise1_rl_debug")
    # Find the first subdirectory
    subdirs = [x for x in base.iterdir() if x.is_dir()]
    if not subdirs:
        print("No subdirectories found!")
        return
        
    img_dir = subdirs[0]
    print(f"Inspecting directory: {img_dir}")
    
    with open("inspect_log.txt", "w") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            
        noisy_path = img_dir / "y_box_noisy.png"
        if noisy_path.exists():
            img = imageio.imread(noisy_path)
            log(f"Noisy Shape: {img.shape}")
            log(f"Noisy Dtype: {img.dtype}")
            log(f"Noisy Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")
        else:
            log(f"Noisy image not found: {noisy_path}")
        
        # Inspect reconstruction
        recon_path = img_dir / "rl" / "rl_box.png"
        if recon_path.exists():
            recon = imageio.imread(recon_path)
            log(f"Recon Shape: {recon.shape}")
            log(f"Recon Min: {recon.min()}, Max: {recon.max()}, Mean: {recon.mean()}")
        else:
            log(f"Recon not found: {recon_path}")

        # Inspect kernel (if saved)
        arrays_dir = img_dir / "arrays"
        if arrays_dir.exists():
            k_path = arrays_dir / "k_box.npy"
            if k_path.exists():
                k = np.load(k_path)
                log(f"Kernel Shape: {k.shape}")
                log(f"Kernel Sum: {k.sum()}")
                log(f"Kernel Min: {k.min()}, Max: {k.max()}")
            else:
                log("Kernel npy not found")
        else:
            log("Arrays dir not found (run with --save-arrays)")

if __name__ == "__main__":
    inspect()
