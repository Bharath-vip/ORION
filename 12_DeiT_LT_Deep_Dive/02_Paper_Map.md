# Paper Map: DeiT-LT (Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets)

## 1. Abstract
The abstract introduces a primary difficulty with Vision Transformers (ViTs): unlike standard Convolutional Neural Networks (CNNs), ViTs do not possess built-in architectural assumptions (like spatial locality). Because of this, they demand enormous amounts of data to train effectively. While recent innovations (like DeiT) have addressed training ViTs on standard, balanced datasets efficiently, the problem of highly imbalanced, "long-tailed" data (where a few classes have many samples and many classes have very few) remains poorly explored for ViTs.

To solve this, the authors propose **DeiT-LT**, a framework designed to train ViTs from scratch on long-tailed data. Their approach brings back knowledge distillation, where a ViT (the student) learns from a CNN (the teacher). By generating out-of-distribution (OOD) images and applying a re-weighted distillation loss, the ViT is explicitly guided to focus on the minority "tail" classes. Furthermore, to prevent overfitting, they utilize a "flat" CNN teacher (trained for smoother loss landscapes). A unique aspect of this framework is the dual-token design: a Distillation (`DIST`) token becomes a specialist for the tail classes, while the standard Classification (`CLS`) token specializes in the majority "head" classes. The authors prove that DeiT-LT succeeds across various imbalanced datasets, ranging from the small-scale CIFAR-10 LT to the large-scale iNaturalist-2018.

## 2. Introduction
In the introduction, the authors highlight the growing dominance of Vision Transformers in computer vision while pointing out their Achilles' heel: the lack of inductive bias. CNNs naturally understand that pixels close to each other are related (locality), whereas ViTs treat image patches equally and must learn these relationships from scratch. When dealing with long-tailed datasets—which mirror real-world data distributions better than perfectly balanced lab datasets—ViTs severely overfit to the majority classes and perform terribly on the minority ones.

The authors note a research gap: while Data-efficient Image Transformers (DeiT) made ViT training viable on balanced datasets without huge pre-training (like JFT-300M or ImageNet-21K), doing the same on imbalanced datasets was an unsolved challenge. Their solution, DeiT-LT, leverages a CNN teacher to inject the missing inductive biases into the ViT. By separating the learning responsibilities into a `CLS` token (for head classes) and a `DIST` token (for tail classes), the model inherently creates two "experts" within the same architecture, drastically improving generalization on both extremes of the data distribution.

## 3. Related Work
This section situates DeiT-LT within three main research streams:
*   **Long-Tailed Recognition (LTR) for CNNs:** Traditional approaches to handle data imbalance include data re-sampling, loss re-weighting (like Focal Loss or Class-Balance Loss), and logit adjustment techniques. The authors build upon these but adapt them for the transformer paradigm.
*   **Vision Transformers & Data Efficiency:** It covers how ViTs traditionally need massive datasets and how distillation-based methods like DeiT-III helped close the gap for balanced datasets.
*   **Long-Tailed ViT Training:** The authors examine existing ViT adaptations for long-tailed data. Importantly, they emphasize that many recent models rely on large-scale multimodal pre-training (like CLIP), which gives them an unfair advantage since they have already seen the "tail" concepts in massive text-image pairs. The authors establish fair baselines by comparing DeiT-LT only against models trained purely from scratch.

## 4. Method
The core framework of DeiT-LT relies on a specialized Knowledge Distillation pipeline to tackle data imbalance.

*   **The Teacher (Flat CNN):** To guide the ViT, the authors employ a Convolutional Neural Network as the teacher. Crucially, they don't just use a standard CNN; they use one trained with Sharpness-Aware Minimization (SAM). SAM encourages the model to find a "flat" minimum in the loss landscape, meaning the learned features are highly generalizable and less prone to overfitting. This "flat" teacher provides robust, low-rank features that the student can easily emulate.
*   **The Student (ViT):** The student is a standard Vision Transformer architecture modified to include a dual-token system. It processes the image patches along with two specialized tokens: the standard `CLS` token and the new `DIST` token.
*   **Distillation Process:** The `DIST` token is responsible for mimicking the outputs of the CNN teacher. Instead of just learning from standard images, the distillation process is heavily enriched by using Out-of-Distribution (OOD) images (created via aggressive data augmentations like Mixup and CutMix). By forcing the `DIST` token to match the CNN teacher's predictions on these highly augmented images, the ViT successfully absorbs the CNN's local inductive biases in its early layers.
*   **Long-tail Strategy:** The true long-tail adaptation happens through a re-weighted distillation loss. The loss function is heavily skewed to prioritize the minority (tail) classes during the distillation process. Because of this targeted guidance, the `DIST` token evolves into a dedicated "Tail Expert." Simultaneously, the standard `CLS` token learns primarily from the standard cross-entropy loss, naturally becoming a "Head Expert." This dual-token division allows the model to handle the entire skewed distribution effectively.

## 5. Experiments
The authors rigorously tested DeiT-LT across a variety of standard long-tailed benchmarks:
*   **Small-scale:** CIFAR-10 LT and CIFAR-100 LT (tested with various imbalance ratios like 10, 50, and 100).
*   **Large-scale:** ImageNet-LT and iNaturalist-2018.

In these experiments, DeiT-LT consistently outperformed standard ViTs trained from scratch, as well as state-of-the-art data-efficient methods like DeiT-III. They demonstrated that their distillation framework provided massive accuracy boosts—particularly on the tail classes—verifying that the `DIST` token successfully specialized as intended.

## 6. Ablation Studies
To prove why their specific design choices matter, the authors performed detailed ablation studies, stripping away components one by one:
*   **OOD Distillation:** They found this to be the single most crucial component. Using highly augmented OOD images for distillation provided massive performance jumps (around 14-18% over the baseline DeiT on CIFAR datasets).
*   **The "Flat" Teacher:** Switching from a standard CNN teacher to a SAM-trained "flat" teacher yielded an additional 3% to 6.7% improvement in accuracy, validating the need for a generalizable teacher.
*   **Deferred Re-weighting (DRW):** They confirmed that dynamically applying the re-weighted distillation loss further refined the tail-expert's capabilities.
Overall, the ablations confirm that every piece of the DeiT-LT puzzle is necessary for optimal performance.

## 7. Discussion
The authors provide a deeper analysis of *how* the model fundamentally works under the hood. Through attention rollout visualizations, they show that the `CLS` and `DIST` tokens actually look at completely different parts of the image. When presented with a tail-class image, the `DIST` token effectively hones in on the most relevant physical features, while the `CLS` token's attention map is scattered and uninformative. This visual evidence perfectly corroborates the "dual-expert" hypothesis, showing that the ViT successfully routes specific class distributions to different tokens.

## 8. Limitations
Despite the strong results, the study points out a few inherent limitations:
*   **Reliance on the Teacher:** The performance of the ViT student is strictly upper-bounded by the quality of the CNN teacher. If the teacher struggles with the data, the student will too. The necessity of pre-training a high-quality, SAM-optimized CNN adds computational overhead.
*   **Inductive Bias Dependency:** The ViT fundamentally still lacks inductive bias; it is merely "borrowing" it from the CNN via distillation. Without this complex, specific setup, the ViT reverts to its data-hungry nature.
*   **Reweighting Constraints:** Relying on standard delayed loss re-weighting (DRW) can sometimes influence primarily the final classifier layers, potentially missing out on encouraging deeper, fine-grained feature separation in the earlier blocks of the transformer.
