# DeiT-LT Kaggle Reproduction Notebook

This folder contains a fully self-contained Kaggle Notebook designed to reproduce the core methodology of **DeiT-LT: Distillation strikes back for Vision Transformer training on Long-Tailed datasets (CVPR 2024)**. 

## 📓 Notebook Information
**File:** `DeiT_LT_Kaggle_IF50.ipynb`

## Initial Reproduction Plan
- [x] Extract core logic (Models, Mixup, DRW, Loss) from official repo
- [x] Handle SAM/PaCo Checkpoint idiosyncrasies (DDP module prefixes, contrastive heads)
- [x] Package into single Kaggle-friendly notebook
- [x] Verify baseline convergence

## Results Tracking

| Model Variation | Epochs | Params | IF | Hardware | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (DeiT-Tiny)** | 300 | 5M | 50 | 2x T4 (Kaggle) | **72.48%** |

*Note: The baseline clearly exhibits the intended DRW behavior, dropping training loss massively at Epoch 270 (0.9 * 300) when mixup is disabled and tail re-weighting activates.*

### Purpose
The goal of this notebook is to provide a "researcher's sandbox" for fast iteration. Instead of dealing with the heavily abstracted multi-file architecture of the official repository, this notebook extracts the essential components into 5 clean, linear cells. This allows you to immediately see how the mathematical improvements (ablations) affect the loss curves and accuracy, without fighting with the codebase structure.

### Notebook Structure
The notebook is divided into 5 clear cells:
1. **Environment Setup**: Installs `timm==0.4.12` and necessary dependencies.
2. **Imports & Weights**: Automatically downloads the official `PaCo SAM ResNet-32` Teacher weights for Imbalance Factor 50 from WandB.
3. **Dataset Generation**: Re-implements the official paper's Pareto distribution logic to generate the exact Long-Tailed CIFAR-10 dataset (IF=50) directly from standard `torchvision` datasets.
4. **Models**: 
   - **Teacher**: A clean PyTorch implementation of `ResNet-32` that parses the contrastive PaCo checkpoint to extract the backbone.
   - **Student**: A 60-line clean-room implementation of `DeiT-Tiny` (5M parameters) equipped with dual experts (CLS token and DIST token).
5. **Main Training Engine**: Contains the exact replication of the paper's optimization schedule:
   - AdamW + Cosine Annealing LR + 5 epochs Warmup.
   - **DRW (Deferred Re-weighting)**: Mixup and Cutmix are active early on, but explicitly disabled in the last 10% of epochs when per-class re-weighting is introduced to boost the Tail Expert.
   - Dual Loss: CrossEntropy for the head expert, Hard Distillation for the tail expert.

### Usage Instructions
1. Upload `DeiT_LT_Kaggle_IF50.ipynb` to [Kaggle](https://www.kaggle.com/).
2. Turn on the **T4 x2** GPU accelerator in the notebook settings.
3. Run All Cells.
4. **Ablation Phase**: To test new hypotheses (e.g., swapping KL Divergence, altering DRW, modifying the tail expert), simply edit the logic in Cell 5 (Main Training Engine) and re-run the cell.

### Hardware Note
The official paper uses `DeiT-Base` (86M parameters) and an effective batch size of 1024, trained over 1200 epochs. To ensure this notebook finishes within Kaggle's 12-hour session limit and facilitates rapid prototyping, it defaults to `DeiT-Tiny` (5M parameters) with a 300-epoch schedule. This serves as the ideal benchmark to measure the relative delta of our future improvements.
