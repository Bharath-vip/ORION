# Common Datasets Used by VAL (Vision and AI Lab) - IISc

## Dataset Distillation
In research such as *Rethinking Dataset Distillation: Hard Truths about Soft Labels*, standard visual datasets are used as baselines to rigorously evaluate soft labels and augmentations:
*   **CIFAR-10 / CIFAR-100**: Often used for initial dataset distillation benchmarks.
*   **TinyImageNet**: A more challenging dataset used to test the scalability of dataset distillation methods.
*   **ImageNet-1K**: The gold standard for scaling dataset distillation techniques to realistic resolutions and class counts.
*   **Long-tailed Datasets**: Custom long-tailed versions of standard datasets (e.g., ImageNet-LT, CIFAR-100-LT) utilized in works like *DeiT-LT*.

## 3D Gaussian Splatting & Radiance Fields
In works like *Turbo-GS* and *ChromaDistill*, high-resolution multi-view datasets are fundamental:
*   **Mip-NeRF 360 Dataset**: The primary benchmark for evaluating state-of-the-art 3D Gaussian Splatting and novel-view synthesis quality.
*   **Tanks and Temples**: Used to evaluate the reconstruction quality in large-scale scenes.
*   **Deep Blending Dataset**: Serves to test high-resolution (often 4K) indoor and outdoor radiance fields.
*   **Synthetic NeRF Datasets**: Often used for baseline debugging and quantitative comparisons against perfect geometry.
