### 1. Classification Loss ($\mathcal{L}_{CE}$)

**1) Purpose:** 
Trains the standard classification (`CLS`) token to correctly predict the target class. In a long-tailed setting, this loss naturally becomes biased towards the head classes because they dominate the dataset frequency.

**2) Variables:**
$$ \mathcal{L}_{CE} = - \sum_{c=1}^{C} y_c \log(\hat{y}_{cls, c}) $$
*   $C$: Total number of classes.
*   $y_c$: Ground truth label (1 if the sample belongs to class $c$, 0 otherwise).
*   $\hat{y}_{cls, c}$: Predicted probability distribution from the `CLS` token output.

**3) Why this equation vs others:** 
This is the standard cross-entropy loss used universally for classification tasks. It is kept unweighted (or weighted only later in training) to allow the ViT to learn general, foundational feature representations from the abundant head classes without disrupting early representation learning.

**4) Implementation:**
```python
import torch.nn.functional as F

# z_cls: Unnormalized logits from the CLS token (Batch, C)
# target: Ground truth class indices (Batch,)
loss_ce = F.cross_entropy(z_cls, target)
```
