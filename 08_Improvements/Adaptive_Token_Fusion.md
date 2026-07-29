# Adaptive Token Fusion (ATF) for DeiT-LT

## The Core Problem
DeiT-LT uses a brute-force 0.5 / 0.5 average between the CLS token and the DIST token during inference. However, paper evidence proves this is suboptimal:
* **CLS Token** is a massive Head Expert (68.3% head acc, but collapses to 13.5% on tail).
* **DIST Token** is a Tail Expert (46.6% tail acc, but weaker 57.2% on head).

By forcing a 50/50 average, the DIST token actively drags down the accuracy on Head classes, causing a -3.8 point regression compared to DeiT-III. 

## The Proposed Solution (ATF)
Instead of a fixed 50/50 split, we propose an **Adaptive Token Fusion** where the fusion weight $\alpha_c$ for each class is a function of its sample frequency $n_c$.

$$ \text{Logits}_{\text{final}}(c) = \alpha_c \cdot \text{Logits}_{\text{CLS}}(c) + (1 - \alpha_c) \cdot \text{Logits}_{\text{DIST}}(c) $$

Where $\alpha_c \to 1.0$ for Head classes, and $\alpha_c \to 0.0$ for Tail classes.

## The Validation Step (Oracle Experiment)
Before attempting to learn a parametric curve (e.g., a 3-parameter spline), we must prove that a per-class optimal $\alpha^*$ actually correlates with class frequency. 

**The Oracle Protocol:**
1. Take a fully trained DeiT-LT model (e.g., our 72.48% baseline).
2. Run the validation set and extract independent predictions for `logits_cls` and `logits_dist`.
3. For each of the 10 CIFAR-10 classes, run a grid search over $\alpha \in [0.0, 1.0]$ in steps of 0.05.
4. Record the $\alpha^*$ that yields the highest accuracy for that specific class.
5. Plot $\alpha^*$ against the class sample count.

If the plot shows a monotonic trend (tail classes prefer lower $\alpha$, head classes prefer higher $\alpha$), ATF is mathematically validated and constitutes a highly publishable, zero-training-cost improvement over DeiT-LT.
