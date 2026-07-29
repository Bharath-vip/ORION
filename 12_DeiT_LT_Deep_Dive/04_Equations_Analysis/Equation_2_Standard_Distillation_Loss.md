### 2. Standard Distillation Loss ($\mathcal{L}_{distill}$)

**1) Purpose:** 
Transfers the inductive biases (like locality and generalizable flat minima) from a pre-trained CNN teacher to the ViT student's Distillation (`DIST`) token.

**2) Variables:**
$$ \mathcal{L}_{distill} = \tau^2 \cdot \text{KL}\left(\sigma\left(\frac{z_{dist}}{\tau}\right) \parallel \sigma\left(\frac{z_{teacher}}{\tau}\right)\right) $$
*   $z_{dist}$: Logit predictions from the student ViT's `DIST` token.
*   $z_{teacher}$: Logit predictions from the CNN teacher.
*   $\tau$: Temperature scaling parameter to soften the probability distributions.
*   $\sigma$: Softmax function.
*   $\text{KL}$: Kullback-Leibler divergence.

**3) Why this equation vs others:** 
Soft distillation via KL Divergence (often preferred over hard label distillation when learning from a CNN) transfers "dark knowledge" — the relative relationships between classes — helping the ViT learn low-rank, robust features from the CNN teacher.

**4) Implementation:**
```python
import torch.nn.functional as F

def standard_distillation_loss(z_dist, z_teacher, tau=1.0):
    log_prob_student = F.log_softmax(z_dist / tau, dim=1)
    prob_teacher = F.softmax(z_teacher / tau, dim=1)
    
    # Calculate KL divergence per sample
    kl_loss = F.kl_div(log_prob_student, prob_teacher, reduction='batchmean')
    return (tau ** 2) * kl_loss
```
