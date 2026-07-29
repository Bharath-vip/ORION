import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import DeiTLT
from teacher import resnet32

def make_long_tail(dataset, num_classes=10, imb_factor=0.01):
    # Pare down the ImageFolder samples to create a Long-Tail distribution
    img_max = len(dataset.samples) / num_classes
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
        
    new_samples = []
    # Group by class
    targets_np = np.array(dataset.targets)
    for the_class in range(num_classes):
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:img_num_per_cls[the_class]]
        for i in selec_idx:
            new_samples.append(dataset.samples[i])
            
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    print(f"Created Long-Tailed dataset with {len(dataset.samples)} total images.")
    print("Class distribution:", img_num_per_cls)
    return dataset

def get_teacher(device):
    teacher = resnet32(num_classes=10)
    url = "https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/3d5f8683cecaf84b2e4130f4f9d1f192/paco_sam_ckpt_cf10_if100.pth.tar"
    weight_path = "teacher_cifar10_lt.pth"
    
    if not os.path.exists(weight_path):
        print("Downloading official ResNet-32 teacher checkpoint...")
        urllib.request.urlretrieve(url, weight_path)
    
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if k.startswith('encoder_q.'):
                new_state_dict[k.replace('encoder_q.', '')] = v
            elif not k.startswith('encoder_k.') and not k.startswith('linear_k.'):
                new_state_dict[k] = v
        teacher.load_state_dict(new_state_dict, strict=False)
    else:
        teacher.load_state_dict(checkpoint, strict=False)
        
    teacher = teacher.to(device)
    teacher.eval() # Teacher is always in eval mode
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

def drw_weight(epoch, total_epochs=300):
    # Deferred Re-Weighting kicks in at 80% of training (epoch 240)
    return 1.0 if epoch < (total_epochs * 0.8) else 2.0

def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    # KL Div loss between Student and Teacher
    log_prob_s = F.log_softmax(student_logits / temperature, dim=1)
    prob_t = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(log_prob_s, prob_t, reduction='batchmean') * (temperature ** 2)
    return loss

def main():
    print("Phase 1: CIFAR-10-LT Fast Iteration Baseline")
    
    batch_size = 64
    num_epochs = 1
    max_steps = 20 # For prototyping the loop
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset setup
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root='./data/cifar10/train', transform=transform)
    lt_dataset = make_long_tail(full_dataset, num_classes=10, imb_factor=0.01) # IF 100
    dataloader = DataLoader(lt_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    
    # 2. Model Initialization (DeiT-Tiny)
    print("Initializing DeiT-Tiny (5M Params) for rapid iteration...")
    model = DeiTLT(num_classes=10, embed_dim=192, depth=12, num_heads=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = GradScaler()
    
    # 3. Teacher Initialization
    teacher = get_teacher(device)
    
    # 4. Training Loop
    model.train()
    print("Starting training...")
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        for i, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            
            with autocast():
                # Teacher Forward
                with torch.no_grad():
                    teacher_logits = teacher(images)
                
                # Student Forward
                logits_cls, logits_dist = model(images)
                
                # Head Expert Loss (Standard Cross Entropy)
                loss_cls = nn.CrossEntropyLoss()(logits_cls, targets)
                
                # Tail Expert Loss (KD from Teacher)
                loss_dist = distillation_loss(logits_dist, teacher_logits)
                
                # Deferred Reweighting applies to the Distillation Loss
                weight = drw_weight(epoch)
                
                loss = loss_cls + (weight * loss_dist)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            if global_step % 5 == 0:
                print(f"Step {global_step} (Processed {global_step*batch_size} images): Loss = {loss.item():.4f} | KD = {loss_dist.item():.4f}")
                
            if global_step >= max_steps:
                print(f"\nCompleted {max_steps} steps. The DeiT-Tiny + Teacher + LT Pipeline is structurally sound!")
                return

if __name__ == "__main__":
    main()
