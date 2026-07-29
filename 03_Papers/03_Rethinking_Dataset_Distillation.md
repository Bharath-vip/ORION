# Rethinking Dataset Distillation: Hard Truths about Soft Labels

**Problem:** Many state-of-the-art large-scale Dataset Distillation (DD) methods show inflated performance metrics due to their reliance on "soft labels" during downstream model training.
**Why it matters:** The community is misjudging the true utility and quality of distilled datasets, conflating the strong regularizing effect of soft labels with actual core data quality.
**Method:** The study systematically benchmarks DD methods against simple random data baselines under strict "hard label" evaluation regimes. It introduces CAD-Prune (a compute-aware pruning metric) and CA2D (a compute-aligned distillation method) for optimized sample selection.
**Dataset:** Standard large-scale datasets frequently used in dataset distillation tasks (e.g., ImageNet subsets).
**Results:** When evaluated fairly without soft labels, many complex DD methods perform no better than random baselines. The introduced CA2D method reliably improves data-efficient learning over existing flawed methods.
**Weaknesses:** Relying strictly on hard labels and compute-aligned pruning adds computational overhead to the dataset distillation preparation phase.
