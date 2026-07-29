# Potential Improvements on Recent Works

1. **Enhancing "AdaptiveSplat" with Semantic Priors:**
   - *Current State:* Allocates Gaussians based on texture awareness.
   - *Improvement:* Integrate a lightweight semantic segmentation foundation model (like MobileSAM) to allocate Gaussians based on object importance and semantics, not just high-frequency texture. This could drastically reduce the number of Gaussians needed for background elements while preserving foreground fidelity.

2. **Dynamic Hard/Soft Label Switching for Dataset Distillation:**
   - *Current State:* "Rethinking Dataset Distillation" exposes hard truths about soft labels.
   - *Improvement:* Propose a curriculum-based distillation approach where the reliance on soft labels dynamically decays during the distillation process, transitioning from soft to hard labels to capture both inter-class relationships initially and strict decision boundaries later.

3. **Motion-Aware Filtering in 4D Gaussians for Extreme Deformation:**
   - *Current State:* "Towards Alias-Free 4D Gaussian Representations" uses motion-aware filtering.
   - *Improvement:* The current filtering might struggle with extreme topological changes (e.g., water splashing, smoke). An improvement would be to introduce a hybrid Eulerian-Lagrangian tracking module to handle fluid or highly non-rigid topological changes in the 4D Gaussian space.
