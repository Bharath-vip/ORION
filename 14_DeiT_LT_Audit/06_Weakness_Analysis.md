# Stage 8: Find the Weakness

### Weakness Identified: Upper-Bound Teacher Dependency

**Evidence:** 
The performance of the `DIST` token (the Tail Expert) relies entirely on the distillation process. Therefore, the ViT student is fundamentally bottlenecked by the capability of the CNN teacher. If the SAM-trained CNN teacher performs poorly on extreme minority classes, the ViT will learn that exact poor performance. The framework doesn't intrinsically solve the long-tail problem on its own; it offloads the problem to a teacher that is assumed to handle it better.

**Experiment Proposal:**
Test the DeiT-LT framework with a deliberately weak teacher (e.g., an under-trained ResNet or one trained without SAM). Measure the degradation of the `DIST` token's performance on the tail classes.

**Expected Result:**
The tail accuracy of the ViT will collapse linearly with the teacher's tail accuracy, proving that DeiT-LT is a "knowledge transfer" framework rather than a pure "long-tailed learning" framework.

**Possible Fix (Research Idea):**
Introduce **Self-Distillation with Memory**. Instead of relying on a static external CNN teacher, the ViT could maintain a moving average (EMA) of its own best historical representations for tail classes in a memory bank. During later epochs (after DRW), it distills knowledge from its own stable historical states (which act as the teacher), removing the need for a separate CNN entirely while maintaining stability.
