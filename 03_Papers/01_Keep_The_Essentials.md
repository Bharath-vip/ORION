# Keep The Essentials: Efficient Reference Conditioned Generation via Token Dropping

**Problem:** Reference-based diffusion models suffer from high computational costs during inference, which scales poorly as the number of reference images increases.
**Why it matters:** Inefficiency severely limits the scalability of multi-reference generation and personalization tasks, making advanced controllable generative AI expensive.
**Method:** The authors introduce a "Token Dropping" (or Sparse Context) strategy. Instead of processing all tokens from references, the model identifies and retains only the essential tokens, filtering out redundant information to reduce the computational burden on the diffusion process.
**Dataset:** Evaluated on standard reference-conditioned text-to-image generation benchmarks.
**Results:** The method significantly reduces runtime overhead and computational cost while preserving the high quality, realism, and spatial controllability of the generated output.
**Weaknesses:** An overly aggressive token dropping threshold could potentially discard fine-grained or subtle details critical to the user's specific reference requirements.
