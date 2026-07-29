# Stage 3: Trace One Batch

This document traces a single image through the DeiT-LT pipeline, identifying the tensor shapes, datatypes, operations, and conceptual memory footprint at each discrete step.

**Scenario:** We are processing a single image (`batch_size=1`) from the CIFAR-10-LT dataset (10 classes) using a `DeiT-Small` model (embedding dimension 384, 12 attention heads, patch size 16x16).

---

## 1. Data Loading & Augmentation
*   **Operation:** Read image, apply CutMix/Mixup, resize to 224x224, normalize.
*   **Tensor Shape:** `(1, 3, 224, 224)`
*   **Datatype:** `torch.float32`
*   **Memory:** ~602 KB
*   **Teacher Tensor:** Usually identical or resized differently depending on config `(1, 3, 224, 224)`.

## 2. Patch Embedding (Patchify)
*   **Operation:** A Conv2d layer with `kernel_size=16`, `stride=16`, `out_channels=384`. It divides the 224x224 image into a 14x14 grid of patches and flattens them into a sequence.
*   **Tensor Shape:** `(1, 384, 14, 14)` -> Flattened to `(1, 196, 384)`
*   **Datatype:** `torch.float32`
*   **Memory:** ~301 KB

## 3. Token Concat & Positional Embedding
*   **Operation:** Prepend the learnable `CLS` token and `DIST` token to the sequence. Then, add the learned 1D positional embeddings.
*   **Tensor Shape:** `(1, 198, 384)`  *(196 patches + 1 CLS + 1 DIST)*
*   **Datatype:** `torch.float32`
*   **Memory:** ~304 KB

## 4. Transformer Blocks (x12)
*   **Operation:** Pass the sequence through 12 identical blocks of Multi-Head Self Attention (MHSA) and Multi-Layer Perceptron (MLP), with Layer Normalization and residual connections.
*   **Tensor Shape:** Remains strictly `(1, 198, 384)` throughout all 12 blocks.
*   **Datatype:** `torch.float32` (or `torch.float16` if using AMP).
*   **Memory Overhead:** Attention matrices `(1, 12, 198, 198)` consume significant VRAM during the forward pass for gradient storage.

## 5. Token Extraction & Classification Heads
*   **Operation:** After the 12th block, slice the tensor to extract the 0th index (CLS) and 1st index (DIST). Pass them through their respective Linear layers to project from 384 dimensions to 10 (number of classes).
*   **Tensor Shape (Extracted):** Two tensors of shape `(1, 384)`
*   **Tensor Shape (Logits):** `z_cls`: `(1, 10)` and `z_dist`: `(1, 10)`
*   **Datatype:** `torch.float32`

## 6. Teacher Forward Pass
*   **Operation:** The CNN teacher processes the same augmented image.
*   **Tensor Shape (Teacher Logits):** `z_teacher`: `(1, 10)`

## 7. Loss Calculation
*   **Operation:**
    1.  **CE Loss:** `CrossEntropy(z_cls, target_label)`
    2.  **KL Loss:** `KLDiv(Softmax(z_dist / tau), Softmax(z_teacher / tau))`
    3.  **DRW Weighting:** Multiply KL Loss by class-specific weight $w_y$ if `epoch >= DRW_epoch`.
*   **Tensor Shape:** Scalar `(1,)` for `loss_ce`, `loss_distill`, and `total_loss`.
*   **Datatype:** `torch.float32`

## 8. Gradient Calculation & Weight Update
*   **Operation:** `total_loss.backward()` computes gradients using chain rule through the stored attention graphs. `optimizer.step()` applies AdamW updates.
*   **Tensor Shape:** Gradients match the shape of every learnable weight in the network.
*   **Memory:** Gradients double the parameter memory footprint (DeiT-Small has ~22M params, so gradients consume ~88MB).
