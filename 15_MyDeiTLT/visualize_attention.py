import torch
import matplotlib.pyplot as plt
import urllib.request
import os
from model import DeiTLT

def main():
    print("Option 2: Inference & Attention Visualization")
    
    # 1. Download weights (Using CIFAR10-LT IF-100 Student weights from README)
    weight_url = "https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/574dab8b51e97a8fca2b88d9f42e53c5/deit_base_distilled_patch16_224_resnet32_1200_CIFAR10LT_imb100_128_%5Bpaco_sam_teacher%5D_best_checkpoint.pth"
    weight_path = "cifar10lt_if100_student.pth"
    
    if not os.path.exists(weight_path):
        print("Downloading official pretrained weights (this may take a minute)...")
        # In a real scenario, uncomment the below to download.
        # urllib.request.urlretrieve(weight_url, weight_path)
        print("Weights would be downloaded here.")
    else:
        print("Weights already downloaded.")
        
    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeiTLT(num_classes=10).to(device)
    model.eval()
    print("Model initialized and set to eval mode.")
    
    # 3. Dummy Image for visualization
    # Shape: 1 image, 3 channels, 224x224
    image = torch.randn(1, 3, 224, 224).to(device)
    
    # 4. Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        logits_cls, logits_dist = model(image)
        
    print(f"CLS Logits shape: {logits_cls.shape}")
    print(f"DIST Logits shape: {logits_dist.shape}")
    
    # 5. Attention Map Visualization Logic
    # (In a full script, you would hook into the TransformerBlock's Attention layer
    #  to extract the attention weights matrix of shape (1, num_heads, 198, 198))
    print("Extracting attention weights from final block (Conceptual)...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title("CLS Token Attention Map (Head Expert)")
    ax2.set_title("DIST Token Attention Map (Tail Expert)")
    
    # Dummy plot to show concept
    ax1.imshow(torch.rand(14, 14).numpy(), cmap='jet')
    ax2.imshow(torch.rand(14, 14).numpy(), cmap='jet')
    
    plt.savefig('attention_maps.png')
    print("Saved visualization to attention_maps.png")

if __name__ == "__main__":
    main()
