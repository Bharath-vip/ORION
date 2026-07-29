import torch
import torch.nn as nn
import torch.nn.functional as F
from distillation import get_drw_weights

class DeiTLTLoss(nn.Module):
    def __init__(self, tau=1.0, alpha=0.5):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        
    def forward(self, logits_cls, logits_dist, logits_teacher, targets, class_frequencies, epoch, drw_epoch):
        # 1. Base Classification Loss (Head Expert)
        loss_ce = F.cross_entropy(logits_cls, targets)
        
        # 2. Distillation Soft Targets (Teacher)
        prob_teacher = F.softmax(logits_teacher / self.tau, dim=1)
        log_prob_student = F.log_softmax(logits_dist / self.tau, dim=1)
        
        # 3. Calculate KL Divergence (Unreduced to apply sample weights)
        kl_loss = F.kl_div(log_prob_student, prob_teacher, reduction='none').sum(dim=1)
        
        # 4. Deferred Re-weighting (Tail Expert Specialization)
        weights = get_drw_weights(class_frequencies, epoch, drw_epoch)
        sample_weights = weights[targets]
        
        loss_distill = (kl_loss * sample_weights).mean() * (self.tau ** 2)
        
        # 5. Total Loss
        total_loss = (1 - self.alpha) * loss_ce + self.alpha * loss_distill
        return total_loss
