# Kontinuous Kontext: Continuous Strength Control for Instruction-based Image Editing

**Problem:** Standard instruction-based image editing models act in a binary fashion, lacking the ability to provide users with fine-grained, continuous control over the intensity or strength of a requested edit.
**Why it matters:** Users often require subtle or partial adjustments rather than extreme transformations, demanding smoother transitions.
**Method:** Builds upon FLUX.1-Kontext by introducing a lightweight projector network. This network maps a continuous scalar control value alongside the text instruction to coefficients in the model's modulation space.
**Dataset:** A specially synthesized and filtered diverse dataset of image-edit-instruction-strength quadruplets.
**Results:** Delivers a smooth, linear decay trend in image similarity as edit strength increases, outperforming basic interpolation methods that suffer from abrupt and unnatural visual transitions.
**Weaknesses:** The method inherits the fundamental limitations of the base FLUX model, which can occasionally struggle with precise geometric edits and complex multi-object relationships.
