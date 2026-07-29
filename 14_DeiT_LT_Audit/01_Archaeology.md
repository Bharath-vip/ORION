# Stage 1: Code Archaeology

This document dissects the core files in the `DeiT-LT` repository, answering the critical questions for reverse-engineering the codebase.

## 1. `main.py` (The entry point, replacing `train.py`)
*   **Why does it exist?** It serves as the primary entry point for training and evaluation. It handles all argument parsing, initialization of the Distributed Data Parallel (DDP) environment, dataset loading, model instantiation (both teacher and student), loss function setup, and the high-level epoch loop.
*   **Who calls it?** The user, via a shell script (e.g., `bash sh/cifar10_imb100.sh`).
*   **Which functions matter?** `main(args)`
*   **What tensor goes in?** None directly (it reads arguments and loads data from disk).
*   **What tensor comes out?** It outputs the final saved model checkpoints (`.pth` files) containing the learned weights.

## 2. `engine.py`
*   **Why does it exist?** It abstracts away the complex training and evaluation loops for a single epoch. It manages the forward pass, loss calculation, backward pass, optimizer steps, and metric logging.
*   **Who calls it?** Called by `main.py` within the epoch loop.
*   **Which functions matter?** `train_one_epoch()` and `evaluate()`.
*   **What tensor goes in?** Batches of images (`samples_student`) and their corresponding labels (`targets`) from the dataloader.
*   **What tensor comes out?** It doesn't return a tensor; it returns a dictionary of aggregated statistics (e.g., `train_stats` containing `loss`, `cls_loss`, `dst_loss`). During the process, it updates the model's weight tensors via `loss.backward()`.

## 3. `deit_models/` (The Student Models)
*   **Why does it exist?** Contains the Vision Transformer architectures. This is where the standard ViT is modified to include the `DIST` (distillation) token alongside the standard `CLS` token.
*   **Who calls it?** Instantiated in `main.py` via `deit_models.__dict__[args.model]`.
*   **Which functions matter?** The `forward_features()` method, which routes the patches, `CLS` token, and `DIST` token through the transformer blocks.
*   **What tensor goes in?** A batch of augmented images: `(B, C, H, W)`.
*   **What tensor comes out?** Typically a tuple of tensors containing the classification logits and distillation logits: `(B, num_classes)`.

## 4. `teacher_models/` (The CNN Teachers)
*   **Why does it exist?** Contains the ResNet and RegNet architectures that act as the teachers. These models possess the strong inductive biases (locality) that the ViT lacks.
*   **Who calls it?** Instantiated in `main.py` if distillation is enabled. 
*   **Which functions matter?** The standard `forward()` pass.
*   **What tensor goes in?** A batch of images: `(B, C, H, W)`. Often, these are clean or differently augmented images compared to the student.
*   **What tensor comes out?** The teacher's logit predictions: `(B, num_classes)`.

## 5. `datasets.py`
*   **Why does it exist?** Handles loading, artificially imbalancing (for long-tailed simulation), and augmenting datasets like CIFAR-LT, ImageNet-LT, and iNaturalist.
*   **Who calls it?** Called by `main.py` to create the dataloaders.
*   **Which functions matter?** `gen_imbalanced_data()` (which enforces the Pareto distribution) and `__getitem__()` (which fetches and augments an image).
*   **What tensor goes in?** Raw image paths/bytes.
*   **What tensor comes out?** A tuple `(student_sample, target)`, where `student_sample` is the transformed image tensor `(C, H, W)`.

## 6. `losses.py`
*   **Why does it exist?** Implements the core logic of DeiT-LT: the Re-weighted Knowledge Distillation loss.
*   **Who calls it?** Instantiated in `main.py` and called during the forward pass in `engine.py`.
*   **Which functions matter?** The `forward()` method of `DistillationLoss`.
*   **What tensor goes in?** The teacher inputs, the student's dual outputs `(outputs_cls, outputs_dist)`, and the ground truth `labels`.
*   **What tensor comes out?** A tuple of scalar loss tensors: `(total_loss, base_loss, distillation_loss)`.

## 7. `sam.py` (Note on SAM)
*   **Why does it exist?** Sharpness-Aware Minimization (SAM) is a specialized optimizer used to train the CNN teachers to find "flat" minima, ensuring they provide highly generalizable features during distillation. 
*   **Note:** While heavily discussed in the paper's theory, the actual `sam.py` optimizer implementation is typically isolated to the teacher-pretraining scripts (often a separate pipeline), which is why we only load the pre-trained weights (`args.teacher_path`) in `main.py` rather than running SAM during the ViT training phase.

## 8. `utils.py`
*   **Why does it exist?** A collection of helper functions for distributed training, checkpoint saving, metric tracking (MetricLogger), and learning rate scaling.
*   **Who calls it?** Almost every other file (`main.py`, `engine.py`).
*   **Which functions matter?** `init_distributed_mode()`, `save_on_master()`, and the `MetricLogger` class.
*   **What tensor goes in/out?** Highly variable depending on the utility function.
