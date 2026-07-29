### 4. Overall Training Objective and Deferred Re-Weighting (DRW)

**1) Purpose:** 
Combines the head-expert (`CLS` token) and tail-expert (`DIST` token) losses. It also implements Deferred Re-Weighting (DRW), where re-weighting is only applied in the later stages of training to prevent damaging the initial learning of visual features.

**2) Variables:**
$$ \mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{rew-distill}(epoch) $$
$$ w_y(epoch) = \begin{cases} 1, & \text{if } epoch < E_{DRW} \\ \frac{1}{n_y}, & \text{if } epoch \ge E_{DRW} \end{cases} $$
*   $\lambda$: Hyperparameter balancing the classification and distillation losses.
*   $E_{DRW}$: The epoch threshold after which the re-weighting is applied.
*   $n_y$: Number of training samples for class $y$.

**3) Why this equation vs others:** 
DRW is a proven heuristic in long-tailed learning. Applying large weights to tail classes from Epoch 1 causes unstable gradients and overfitting because the model hasn't learned basic shapes/textures yet. Deferring the weights allows the ViT to learn rich representations first, then specialize its `DIST` token later.

**4) Implementation:**
```python
import torch
import torch.nn.functional as F

def deit_lt_total_loss(z_cls, z_dist, z_teacher, target, class_frequencies, epoch, drw_epoch=160, lambda_weight=1.0):
    # 1. Standard CE Loss
    loss_ce = F.cross_entropy(z_cls, target)
    
    # 2. Compute weights based on DRW schedule
    if epoch < drw_epoch:
        # Uniform weights before DRW epoch
        weights = torch.ones_like(class_frequencies, dtype=torch.float32)
    else:
        # Inverse frequency weighting after DRW epoch
        weights = 1.0 / class_frequencies
        # Normalize weights so they sum to the number of classes (optional but standard)
        weights = weights / weights.sum() * len(class_frequencies)
        
    # 3. Compute Re-weighted Distillation Loss
    # (Assuming reweighted_distillation_loss is defined as in Eq 3)
    loss_distill = reweighted_distillation_loss(z_dist, z_teacher, target, weights)
    
    # 4. Total Loss Combination
    return loss_ce + lambda_weight * loss_distill
```
