# Poster Presentation Script: Physics-Aware Deblurring with Coded Exposure

**Time Estimate:** ~5 Minutes
**Speakers:** Pedram Yazdinia, Seo Won Yi

---

## 1. Introduction & Motivation (0:00 - 0:45)

**[Point to Title & Authors]**
"Hi everyone, I'm [Name]. Our work is on **Physics-Aware Deblurring with Coded Exposure**."

**[Point to 'Motivation' Section]**
"We're motivated by a common problem in robotics, drones, and mobile photography: **Motion Blur**. When a camera moves during exposure, you lose high-frequency details, making the image unusable for downstream tasks like SLAM or object detection."

"Standard cameras leave the shutter open, resulting in a 'Box' blur. In the frequency domain, this is a disaster.
**[Point to the MTF Plot - Blue Line]**
Look at this blue line. This is the **Modulation Transfer Function (MTF)** of a standard shutter. It has multiple 'zeros'—frequencies that are completely destroyed. No algorithm can recover information that is mathematically zeroed out."

"To fix this, we use **Coded Exposure**. By fluttering the shutter open and closed during integration, we can shape the MTF.
**[Point to the MTF Plot - Green Line]**
The green line shows a **Legendre** code. Notice how it avoids those zeros? It preserves broadband frequency information, making deblurring actually possible."

---

## 2. The Problem with Existing Methods (0:45 - 1:30)

"So, we have good optics (Coded Exposure). How do we solve the inverse problem?"

**[Point to 'Related Work' or transition to 'New Technique']**
"Classically, people used **Wiener Deconvolution** or **Richardson-Lucy**. These are fast but struggle with noise and artifacts."

"More recently, **Plug-and-Play (PnP) ADMM** has become the gold standard. It splits the problem into two steps:
1.  **Inversion**: Inverting the blur kernel.
2.  **Denoising**: Using a deep neural network (like a UNet or DRUNet) as a prior to clean up the result."

"**However, there's a gap.** Standard PnP algorithms are 'physics-blind'. They treat the data consistency step as a black box. They don't account for the fact that *this* specific exposure code has *strong* signal at some frequencies and *weak* signal at others. They try to enforce consistency equally everywhere, which amplifies noise in those low-MTF regions."

---

## 3. Our Method: Physics-Aware PnP (1:30 - 3:00)

**[Point to 'New Technique' / Central Diagram]**
"This is where our contribution comes in. We inject **'Optical Intelligence'** into the PnP framework."

"We introduce three key innovations:"

**A. Physics-Aware Weighting (The Core Novelty)**
**[Point to 'Trust Weights (W)' block in diagram]**
"We pre-compute a frequency-domain **'Trust Map' (W)** based on the kernel's MTF.
*   If the MTF is high (good signal), $W$ is close to 1. We trust the measurement.
*   If the MTF is low (noise dominated), $W$ drops. We trust the **Deep Prior** instead."

**B. MTF-Selective Data Consistency**
**[Point to '1. MTF-Aware X-Update' block]**
"We modify the standard ADMM inversion step. Instead of a standard least-squares update, we use a **Weighted Least Squares** update using our Trust Map. This prevents the algorithm from overfitting to noise in the frequency nulls."

**C. Adaptive Physics Scheduler**
**[Point to 'Scheduler' block at bottom of diagram]**
"Standard ADMM requires painful manual tuning of the penalty parameter $\rho$. We built an **Adaptive Scheduler** that dynamically adjusts $\rho$ and the denoiser strength at every iteration based on the current signal-to-noise ratio estimate. It effectively 'anneals' the problem from coarse to fine details automatically."

---

## 4. Experimental Results (3:00 - 4:15)

**[Point to 'Experimental Results' Table]**
"Let's look at the numbers. We compared our method against **Richardson-Lucy** and standard **ADMM with Deep Priors (DnCNN, DRUNet)**."

"Across all shutter codes (Box, Random, Legendre), our **Physics-Aware** approach (bottom row) consistently outperforms the baselines."
*   "For the **Legendre** code, we achieve a **29.27 dB PSNR**, which is a massive **~5 dB gain** over standard ADMM with DnCNN."
*   "Even against the strong **DRUNet** baseline, we see significant improvements because we are handling the frequency domain more intelligently."

**[Point to Visual Crops]**
"Visually, you can see the difference here in the butterfly and bookshelf crops.
*   **Top Row (Blur+Noise)**: Heavily degraded.
*   **Bottom Row (Reconstruction)**: Our method recovers sharp edges and fine textures without the ringing artifacts typical of Wiener filters or the over-smoothing of standard deep priors."

---

## 5. Conclusion (4:15 - 5:00)

"In summary, we show that **optics and algorithms cannot be designed in isolation**."

"By making the reconstruction algorithm explicitly aware of the **MTF properties** of the exposure code, we get the best of both worlds:
1.  The **mathematical invertibility** of Coded Exposure.
2.  The **generative power** of Deep Learning priors.
3.  The **robustness** of Physics-Aware optimization."

"Thank you! We're happy to take any questions."
