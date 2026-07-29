# Stage 7: Question Every Assumption

A true researcher doesn't just accept the methodology; they question the "why" behind every design choice.

### 1. Why a CNN teacher?
ViTs lack inductive biases (translation invariance, locality). CNNs naturally possess these. Distilling from a CNN acts as a rapid "injection" of these priors into the ViT, saving it from needing hundreds of millions of pre-training images.

### 2. Why SAM (Sharpness-Aware Minimization)?
Standard CNNs can overfit to specific high-frequency details. SAM explicitly optimizes for "flat minima" in the loss landscape. A flat minimum means the teacher's features are highly generalizable and robust. When the ViT mimics a SAM-teacher, it inherently inherits this generalization, drastically preventing overfitting on the tail classes.

### 3. Why DRW (Deferred Re-Weighting)?
If you heavily weight the tail classes from Epoch 1, the model's gradients become highly unstable. The network is essentially being screamed at to classify a rare object before it even knows what an edge or a color gradient is. DRW allows the model to learn basic visual representations neutrally first, and then specializes the classifier later.

### 4. Why KL Divergence for Soft Targets?
KL Divergence compares probability distributions. It transfers "dark knowledge" (e.g., a cat is more similar to a dog than to a car). This gives the student significantly more structural information than just binary 1/0 hard labels.

### 5. Why OOD (Out-Of-Distribution) images?
If you only distill on normal images, the student just copies the teacher's specific answers. By feeding highly corrupted or mixed images (OOD), you force the student to learn *how the teacher thinks* when confused, pushing the student to adopt the teacher's internal logic and robust feature mappings.

### 6. Why Mixup / CutMix?
They are computationally cheap ways to generate infinite OOD images and prevent memorization. They act as strong regularizers.

### 7. Why CLS and DIST separate tokens?
Attempting to make one token handle both the massive head classes and the sparse tail classes leads to gradient conflicts (the head dominates). Splitting them allows for specialized "experts."

### 8. Why ViT-Small? Why not ViT-Base?
ViT-Small has ~22M parameters, comparable to a ResNet-50. It's the standard baseline for data-efficient research. ViT-Base (86M params) would overfit instantly on small long-tailed datasets without massive ImageNet-21K pretraining, defeating the purpose of the study.
