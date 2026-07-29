# Do Vision Language Models Need to Process Image Tokens?

**Problem:** Vision Language Models (VLMs) traditionally feed dense image tokens throughout all deep transformer layers, consuming massive computational resources.
**Why it matters:** Questioning this architectural assumption opens pathways to significantly optimize VLM efficiency via pruning, without compromising model reasoning or output quality.
**Method:** The researchers analyze the evolution of visual representations across layers, finding they quickly converge to a "bounded-complexity regime." Based on this, they evaluate the impact of truncating visual token processing at varying depths.
**Dataset:** Evaluated on multiple VLM benchmarks encompassing single-token prediction and multi-token reasoning tasks.
**Results:** Early-layer processing of visual tokens is often sufficient. Single-token tasks remain robust to depth truncation, while complex multi-token tasks only need sustained processing up to a point, indicating deep visual layers are often redundant.
**Weaknesses:** Truncating visual depth can perturb intermediate reasoning trajectories in deterministic decoding, meaning the reasoning process itself may become less structured even if the final answer is correct.
