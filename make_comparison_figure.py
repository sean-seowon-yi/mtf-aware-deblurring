import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from pathlib import Path

def load_image(path):
    if not path.exists():
        print(f"Warning: {path} not found")
        return np.zeros((256, 256, 3), dtype=np.uint8)
    return imageio.imread(path)

def make_figure(regime_name, vanilla_dir, physics_dir, output_path, image_ids=["0001x2", "0002x2"], code="legendre"):
    rows = len(image_ids)
    cols = 4 # GT, Noisy, Vanilla, Physics
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes[None, :]
        
    titles = ["Ground Truth", f"Noisy ({regime_name})", "Vanilla ADMM", "Physics-Aware ADMM"]
    
    for i, img_id in enumerate(image_ids):
        # Paths
        # GT is scene.png
        gt_path = vanilla_dir / img_id / "scene.png"
        # Noisy is y_{code}_noisy.png
        noisy_path = vanilla_dir / img_id / f"y_{code}_noisy.png"
        
        # Recons are saved in the method folder
        vanilla_recon_path = vanilla_dir / img_id / "admm" / f"admm_{code}.png"
        physics_recon_path = physics_dir / img_id / "admm" / f"admm_{code}.png"
        
        # Load
        gt = load_image(gt_path)
        noisy = load_image(noisy_path)
        vanilla = load_image(vanilla_recon_path)
        physics = load_image(physics_recon_path)
        
        imgs = [gt, noisy, vanilla, physics]
        
        for j, ax in enumerate(axes[i]):
            ax.imshow(imgs[j])
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[j], fontsize=14, fontweight='bold')
            if j == 0:
                ax.text(-0.1, 0.5, f"Image {img_id}", transform=ax.transAxes, 
                        rotation=90, va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

def main():
    base = Path("report/comparisons")
    figs_out = Path("report/figures")
    figs_out.mkdir(exist_ok=True, parents=True)
    
    # Noise 1
    make_figure(
        "Noise1",
        base / "noise1_vanilla",
        base / "noise1_physics",
        figs_out / "comparison_noise1.png"
    )
    
    # Noise 2
    make_figure(
        "Noise2",
        base / "noise2_vanilla",
        base / "noise2_physics",
        figs_out / "comparison_noise2.png"
    )

if __name__ == "__main__":
    main()
