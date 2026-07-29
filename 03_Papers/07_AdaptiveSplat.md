# AdaptiveSplat: Texture Aware Controllable 3D Gaussian Allocation for Feed-Forward Reconstruction

**Problem:** Feed-forward 3D Gaussian Splatting (3DGS) often generates highly redundant, pixel-aligned Gaussian primitives. Traditional pruning removes these but causes artifacts, breaking the efficient feed-forward paradigm.
**Why it matters:** Minimizing Gaussian count without requiring slow, post-processing fine-tuning is vital for high-speed, efficient 3D reconstruction and rendering.
**Method:** Introduces an adaptive allocation strategy driven by local texture information. It intelligently concentrates Gaussian primitives in high-frequency, detailed regions while sparing them in low-texture or flat areas.
**Dataset:** Evaluated on standard 3DGS benchmarks and real-world multi-view datasets.
**Results:** Significantly reduces redundancy and improves overall reconstruction efficiency natively, completely avoiding the need for inference-time optimization or fine-tuning.
**Weaknesses:** Heavy reliance on texture cues might cause the model to under-allocate primitives in areas with strong specular highlights on physically smooth, untextured surfaces.
