# DeiT-LT: Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets

### 1. What problem are they solving?
The paper addresses the challenge of training Vision Transformers (ViTs) from scratch on **long-tailed (imbalanced) datasets**. In these datasets, a few majority classes have a large number of samples, while many minority ("tail") classes have very few samples.

### 2. Why is it important?
Vision Transformers have emerged as powerful architectures for computer vision tasks, but they are notoriously "data-hungry" because they lack the built-in inductive biases (like translation invariance and spatial locality) found in Convolutional Neural Networks (CNNs). While there are efficient training methods for balanced datasets (like DeiT), real-world data naturally exhibits a long-tailed distribution. Enabling ViTs to be trained effectively on long-tailed datasets directly—without requiring massive large-scale pre-training—is crucial for their practical application in real-world scenarios.

### 3. Why do existing methods fail?
Existing methods for training ViTs from scratch (such as the original DeiT) are designed for balanced datasets. Because ViTs lack inherent inductive biases, they rely on massive amounts of data to learn spatial structures and local features from scratch. In a long-tailed setting, the minority "tail" classes do not provide enough examples for the ViT to learn these fundamental properties. Consequently, the model struggles to generalize and performs poorly on underrepresented classes.

### 4. What is their main idea?
The core idea is an **Out-of-Distribution (OOD) distillation framework** called **DeiT-LT**. The authors use a flat Convolutional Neural Network (CNN) trained with Sharpness-Aware Minimization (SAM) as a "teacher" model to guide the ViT "student". By distilling knowledge from the CNN teacher—which inherently possesses strong inductive biases—into the ViT, the framework encourages the ViT to learn robust, local, CNN-like features in its early layers. They also employ a dual-expert strategy, utilizing a distillation (`DIST`) token and a class (`CLS`) token to specialize in tail and head classes respectively. 

### 5. What are their contributions?
- **DeiT-LT Framework:** Introducing the first effective OOD distillation framework specifically designed for training ViTs from scratch on long-tailed datasets.
- **Dual-Expert Distillation Strategy:** Utilizing the standard `CLS` token and the distillation `DIST` token to effectively act as a "head expert" and "tail expert", allowing the model to handle majority and minority classes simultaneously.
- **Injecting Inductive Biases:** Demonstrating that a CNN teacher trained via SAM can successfully impart beneficial inductive biases (like local feature learning) into a ViT student.
- **Reweighted OOD Distillation:** Employing OOD images generated via augmentations (e.g., Mixup, CutMix) along with a specialized re-weighted distillation loss that heavily prioritizes and boosts the performance of the neglected tail classes.
