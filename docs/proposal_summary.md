# Proposal Summary

## Motivation
- Motion blur under low light or long exposure causes frequency nulls that ill-condition deblurring.
- Coded exposure (flutter shutter) redistributes spectral energy to avoid deep MTF zeros but still amplifies noise.
- Modern deep priors (PnP, diffusion) ignore system physics when choosing denoising schedules.
- Project goal: adapt denoising strength to the measured/simulated MTF to stabilize inversions without retraining.

## Related Work Highlights
- Flutter shutter coding (Raskar et al., 2006) and Modified Legendre Sequences (Jeon et al., 2013) improve conditioning at capture time.
- Plug-and-Play priors with deep denoisers (DPIR) and diffusion-based variants provide strong reconstructions but use fixed schedules.
- Opportunity: blend physics-based capture with adaptive prior control to reduce ringing/noise.

## Project Objectives
- Demonstrate that an MTF-aware denoising schedule inside a PnP/HQS loop improves stability and perceptual quality versus fixed/monotone schedules.
- Compare box versus coded exposures under equal photon budgets with Poisson-Gaussian noise.
- Release reproducible code, configs, and a polished report suitable for course deliverables.

## Planned Approach
- **Data**: Synthetic scenes blurred with chosen flutter codes; optional real capture of a controlled target.
- **Methods**: Coded forward model `y = k * x + n`, PnP/HQS with denoiser whose schedule s(f) depends on |OTF(k)|; baselines include Wiener, Richardson-Lucy, fixed schedules, and different codes.
- **Evaluation**: PSNR/SSIM/LPIPS, spectral SNR analysis, qualitative crops, robustness to code patterns and photon budgets.
- **Ethics**: Synthetic or benign lab data only; release code/params.
- **Deliverables**: Reproducible repo, report with plots/tables, optional demo video.

## Milestones & Timeline
- **By Oct 24**: Stand up simulator, Poisson-Gaussian noise, PSF/OTF/MTF sanity plots.
- **By Oct 31**: Implement Wiener and Richardson-Lucy baselines, PnP/HQS with fixed s.
- **By Nov 7**: Integrate first physics-aware s schedule, obtain qualitative results.
- **By Nov 14**: Run ablations across schedules and code patterns, report spectral SNR and LPIPS.
- **By Nov 21**: Stress-test robustness to photon budget, motion, noise; prep figures.
- **By Nov 28** *(optional)*: Capture tiny real demo and process coded image.
- **By Dec 5**: Final report, repo polish, reproducibility checklist, slide-ready summary.
