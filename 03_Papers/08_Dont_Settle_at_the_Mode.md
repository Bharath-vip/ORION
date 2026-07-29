# Don't Settle at the Mode! Mitigating Diversity Collapse in Pretrained Flow Models via Feature Self-Guidance

**Problem:** Pretrained flow-based generative models often suffer from "diversity collapse," producing visually similar outputs for the same conditioning prompt despite sampling variations.
**Why it matters:** High-quality generative AI must provide a diverse array of interpretations for user prompts without sacrificing fidelity or slowing down inference time.
**Method:** Identifies a link between internal feature collapse and output collapse. Proposes a training-free feature self-guidance mechanism with manifold regularization to disperse internal features while anchoring them to the valid data distribution.
**Dataset:** Evaluated on standard text-to-image generation benchmarks for diversity and quality.
**Results:** Outperforms existing baseline diversity-enhancement techniques. It successfully boosts sample variety while preserving prompt adherence and the high-speed inference inherent to flow models.
**Weaknesses:** Because it is applied at inference time, finding the optimal feature dispersion hyperparameters might require domain-specific tuning to balance diversity against quality degradation.
