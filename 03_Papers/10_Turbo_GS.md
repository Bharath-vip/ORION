# Turbo-GS: Accelerating 3D Gaussian Fitting for High-Resolution Radiance Fields

**Problem:** Standard 3D Gaussian Splatting (3DGS) requires a notoriously slow training (fitting) process, especially for high-resolution (4K) radiance fields, often taking hours.
**Why it matters:** Drastically accelerating the optimization step is necessary to make high-resolution 3DGS practical for rapid asset creation and real-world deployment.
**Method:** Employs dilated rendering (rendering pixel subsets), an effective densification strategy combining positional and appearance errors, and a convergence-aware dynamic budget control to prevent oversaturation of Gaussians.
**Dataset:** Tested on high-resolution 4K radiance field benchmarks.
**Results:** Reduces optimization steps to roughly one-third of traditional 3DGS. High-resolution scenes converge in minutes rather than hours, maintaining or exceeding the visual fidelity of SOTA baselines.
**Weaknesses:** Dilated rendering inherently samples fewer pixels per step, which could theoretically cause the model to miss extremely thin or sub-pixel structures during the accelerated fitting process.
