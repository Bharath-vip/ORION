# MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World

**Problem:** Modern text-to-image diffusion models consistently fail to generate physically accurate mirror reflections, shadows, and occlusions in complex real-world scenes.
**Why it matters:** Accurate reflections are essential for achieving true photorealism in synthetic media, with significant implications for advanced image editing, AR, and VR.
**Method:** Frames reflection generation as an inpainting task. Uses depth-conditioning and a mask for the mirror region, training a new model (MirrorFusion 2.0) via a specialized three-stage curriculum learning strategy.
**Dataset:** SynMirrorV2, an enhanced dataset of 207,000 synthetic samples featuring randomized rotations, multi-object complexity, and varied backgrounds.
**Results:** Establishes a new state-of-the-art for depth-conditioned mirror reflection generation, successfully managing complex spatial layouts and occlusions far better than prior models.
**Weaknesses:** Training still relies heavily on synthetic data, which may not completely bridge the gap to highly irregular or distorted real-world mirror shapes.
