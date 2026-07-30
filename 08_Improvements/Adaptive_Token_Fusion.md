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

## Successful Validation (July 2026)
The Oracle experiment was successfully run on Kaggle using a fully trained DeiT-Tiny on CIFAR-10 LT (IF=50). The results perfectly validated the ATF hypothesis:

| Class | Count | Group | Best Oracle $\alpha^*$ |
| :--- | :--- | :--- | :--- |
| 0 | 5000 | Head | 0.80 |
| 1 | 3237 | Head | 1.00 |
| 2 | 2096 | Head | 1.00 |
| 3 | 1357 | Head | 1.00 |
| 4 | 878 | Med | 0.05 |
| 5 | 568 | Med | 0.05 |
| 6 | 368 | Med | 0.00 |
| 7 | 238 | Tail | 0.05 |
| 8 | 154 | Tail | 0.00 |
| 9 | 100 | Tail | 0.00 |

**Conclusion:** The optimal fusion weight follows an almost perfect step-function correlated with class frequency. Head classes demand $\alpha \approx 1.0$ (relying exclusively on CLS), while Medium and Tail classes demand $\alpha \approx 0.0$ (relying exclusively on DIST). 

Even with a flat $\alpha \approx 0.26$ bias learned by the initial Scipy optimizer, the model achieved a **+1.18% absolute accuracy improvement** (73.22% $\to$ 74.40%) at zero extra training cost.

## Next Steps (Scipy Optimizer Fix)
During the first ATF experiment, the Scipy optimizer learned a flat curve (`w1=0.0`, `w2=0.0`, `b=-1.0`). This occurred because the L2 regularization penalty (`0.1 * (w1**2 + w2**2)`) was excessively large compared to the CrossEntropy loss, forcing the weights to zero. 

**Action Item:** In the next iteration, remove or drastically reduce the L2 regularization penalty in the `atf_loss` function to allow the optimizer to learn the true diagonal curve demonstrated by the Oracle plot. This should unlock the full theoretical limit of the ATF boost.
