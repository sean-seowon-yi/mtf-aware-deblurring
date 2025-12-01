import torch
from pathlib import Path
from mtf_aware_deblurring.denoisers.tiny_denoiser import default_denoiser_weights

def check_weights():
    path = default_denoiser_weights()
    print(f"Loading weights from: {path}")
    ckpt = torch.load(path, map_location="cpu")
    
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
        
    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    check_weights()
