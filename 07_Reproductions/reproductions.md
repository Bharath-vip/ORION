# Potential Reproduction Targets

1. **Rethinking Dataset Distillation: Hard Truths about Soft Labels (CVPR 2026)**
   - *Why:* Dataset distillation is computationally heavy. Reproducing this on a smaller scale (e.g., CIFAR-100 or TinyImageNet) to verify the claims regarding soft labels will provide a strong foundation for future efficient-learning projects.
   - *Target:* Re-implement the distillation loop and evaluate the exact performance drop when soft labels are replaced or hardened.

2. **AdaptiveSplat: Texture Aware Controllable 3D Gaussian Allocation (ECCV 2026)**
   - *Why:* 3D Gaussian Splatting is a highly active field. Reproducing the feed-forward reconstruction pipeline will yield deep insights into Gaussian allocation strategies.
   - *Target:* Implement the texture-aware allocation module on top of a baseline 3D-GS codebase and test it on custom scenes to evaluate rendering speed and PSNR improvements.

3. **Feature Self-Guidance in Pretrained Flow Models (ECCV 2026)**
   - *Why:* Flow matching and flow models are replacing standard diffusion in many domains.
   - *Target:* Apply their feature self-guidance technique to a small, publicly available pretrained flow model (e.g., on CelebA or a toy 2D dataset) to visually verify the mitigation of diversity collapse.
