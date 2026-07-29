### 3. Re-weighted Distillation Loss ($\mathcal{L}_{rew-distill}$)

**1) Purpose:** 
Forces the `DIST` token to become an expert on the **tail classes**. By heavily weighting the distillation loss for minority classes, the token shifts its focus away from the head classes.

**2) Variables:**
$$ \mathcal{L}_{rew-distill} = w_y \cdot \mathcal{L}_{distill} $$
*   $w_y$: The class-specific re-weighting factor for the target class $y$. This is typically inversely proportional to the class frequency $n_y$ (e.g., $w_y \propto 1/n_y$ or the effective number of samples).

**3) Why this equation vs others:** 
Standard distillation would simply replicate the teacher's predictions (including its bias towards head classes). Re-weighting specifically the distillation loss (and not the main CE loss initially) isolates the tail-class specialization to the `DIST` token, preserving the overall representation quality.

**4) Implementation:**
```python
import torch.nn.functional as F

def reweighted_distillation_loss(z_dist, z_teacher, target, weights, tau=1.0):
    log_prob_student = F.log_softmax(z_dist / tau, dim=1)
    prob_teacher = F.softmax(z_teacher / tau, dim=1)
    
    # Calculate KL divergence per sample without reducing
    kl_loss = F.kl_div(log_prob_student, prob_teacher, reduction='none').sum(dim=1)
    
    # Apply class-specific weights based on the ground truth target
    sample_weights = weights[target]
    weighted_loss = (kl_loss * sample_weights).mean()
    
    return (tau ** 2) * weighted_loss
```
