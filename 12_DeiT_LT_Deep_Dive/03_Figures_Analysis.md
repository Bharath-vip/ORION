# Analysis of Figures in "DeiT-LT: Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets"

### Figure 1
**1) Why is it here?**
To provide a high-level conceptual overview of the DeiT-LT framework tailored for long-tailed data.
**2) What does it prove?**
It demonstrates the core hypothesis: distilling from a CNN using Out-Of-Distribution (OOD) images helps early ViT blocks learn local, generalizable features, and using Sharpness-Aware Minimization (SAM) teachers induces low-rank features that improve generalization.
**3) Why did the authors draw it this way?**
To present an abstract, easy-to-understand visual summary that contrasts their novel approach (OOD distillation + SAM teacher) with standard ViT training baselines.
**4) Could I redraw it myself?**
Yes. You could recreate it using diagramming tools like draw.io or PowerPoint by drawing flowchart blocks that connect a CNN teacher to a ViT student, highlighting the OOD data input and SAM-based training concepts.

### Figure 2
**1) Why is it here?**
To act as a detailed architectural and functional diagram of the proposed DeiT-LT distillation scheme.
**2) What does it prove?**
It illustrates the mechanics of the dual-expert system: it proves how the model integrates a `DIST` (distillation) token that learns to become an expert on tail classes (via re-weighted distillation loss) while the `CLS` (classification) token acts as an expert for head classes.
**3) Why did the authors draw it this way?**
To clearly differentiate the processing pathways and loss functions applied to the `CLS` and `DIST` tokens, making the integration of deferred re-weighting structurally apparent.
**4) Could I redraw it myself?**
Yes. It can be redrawn using tools like TikZ, draw.io, or Excalidraw by mapping out the ViT layers, drawing parallel paths for the CLS/DIST tokens, and labeling the respective loss functions.

### Figure 3
**1) Why is it here?**
To further detail the expert specialization and architectural token design within the transformer blocks.
**2) What does it prove?**
It shows the functional separation between the classification and distillation tokens, illustrating how they operate as specialized experts for different segments of the data distribution (head vs. tail).
**3) Why did the authors draw it this way?**
To visually reinforce the complementary roles of the two tokens and how they aggregate information distinctly across the transformer layers.
**4) Could I redraw it myself?**
Yes, by creating a block diagram focusing on the transformer's output heads and the specific routing of the CLS and DIST tokens.

### Figure 4
**1) Why is it here?**
To quantitatively and mechanistically illustrate the effect of distillation in the DeiT-LT framework.
**2) What does it prove?**
Part (a) proves that OOD distillation with deferred re-weighting (DRW) creates more diverse experts compared to in-distribution training. Part (b) proves that without OOD distillation, standard ViT baselines overfit to spurious global features (evidenced by the "Mean Attention Distance" across early self-attention blocks).
**3) Why did the authors draw it this way?**
Using a combination of bar/line charts and attention distance plots allows the authors to provide both performance metrics and internal mechanistic insights in a single cohesive view.
**4) Could I redraw it myself?**
Yes, using Python plotting libraries like Matplotlib or Seaborn, provided you have the raw training metrics and the calculated mean attention distances per transformer block.

### Figure 5
**1) Why is it here?**
To provide a qualitative visual comparison of the attention maps for images from tail classes.
**2) What does it prove?**
It proves visually that the `DIST` token effectively focuses on meaningful semantic regions of tail class images, whereas the `CLS` token struggles or focuses on spurious background features. This validates the dual-expert hypothesis.
**3) Why did the authors draw it this way?**
Overlaying attention heatmaps (e.g., using Attention Rollout) directly onto the original images is the standard and most intuitive way to show exactly where the network is "looking."
**4) Could I redraw it myself?**
Yes, if you have the trained DeiT-LT model weights, you could use a PyTorch script to extract attention maps via Attention Rollout and overlay them onto sample images using OpenCV or Matplotlib.

### Figure S.4 (Supplementary)
**1) Why is it here?**
To analyze the feature representations learned by the CLS and DIST tokens on the CIFAR-10 LT dataset.
**2) What does it prove?**
It proves that DeiT-LT captures fine-grained features (indicated by a high-rank CLS token) and generalizable features (indicated by a low-rank DIST token), supporting the theoretical claims of the paper.
**3) Why did the authors draw it this way?**
Using rank comparison plots effectively quantifies abstract theoretical claims regarding feature dimensionality and generalization.
**4) Could I redraw it myself?**
Yes, by extracting the feature matrices of both tokens from the model and plotting their singular value spectrums using NumPy and Matplotlib.

### Figure S.5 (Supplementary)
**1) Why is it here?**
To present a convergence analysis for different model variants (e.g., with and without SAM teachers).
**2) What does it prove?**
It demonstrates the impact of using Sharpness-Aware Minimization (SAM) teachers on the training speed and final convergence stability on long-tailed datasets like ImageNet-LT and CIFAR-100 LT.
**3) Why did the authors draw it this way?**
Line graphs mapping loss/accuracy against epochs are the standard, most readable format for comparing convergence behaviors over time.
**4) Could I redraw it myself?**
Yes, by logging the training and validation metrics during a training run and plotting them with a standard charting library.
