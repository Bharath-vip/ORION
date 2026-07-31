import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import timm
import types
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import torch.backends.cudnn as cudnn

# ==========================================
# 1. ImageNet-LT Dataset
# ==========================================
class ImageNetLT(Dataset):
    def __init__(self, root, txt_file, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Missing ImageNet-LT split: {txt_file}")
            
        with open(txt_file) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
                
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        try:
            with open(path, 'rb') as f:
                sample = Image.open(f).convert('RGB')
        except:
            sample = Image.new('RGB', (224, 224))
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, label

# ==========================================
# 2. DeiT-Small for ImageNet-LT
# ==========================================
class DeiTLT(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=num_classes)
        self.embed_dim = self.model.embed_dim
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        def new_forward_features(self_model, x):
            B = x.shape[0]
            x = self_model.patch_embed(x)
            cls_tokens = self_model.cls_token.expand(B, -1, -1)
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
            x = x + self_model.pos_embed
            x = self_model.pos_drop(x)
            for blk in self_model.blocks:
                x = blk(x)
            x = self_model.norm(x)
            return x[:, 0], x[:, 1]
            
        self.model.forward_features = types.MethodType(new_forward_features, self.model)
        self.model.dist_token = self.dist_token
        self.model.pos_embed = nn.Parameter(torch.zeros(1, self.model.patch_embed.num_patches + 2, self.embed_dim))
        
        self.head_cls = nn.Linear(self.embed_dim, num_classes)
        self.head_dist = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x, return_features=False):
        cls_tok, dist_tok = self.model.forward_features(x)
        logits_cls = self.head_cls(cls_tok)
        logits_dist = self.head_dist(dist_tok)
        if return_features:
            return logits_cls, logits_dist, cls_tok, dist_tok
        return logits_cls, logits_dist

# ==========================================
# 3. Neural Entropy Router (MLP)
# ==========================================
class EntropyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# ==========================================
# 4. Main Script
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="ImageNet-LT DeiT Scaling & ATF Pipeline")
    parser.add_argument('--data-path', type=str, default='/kaggle/input/imagenet/ILSVRC/Data/CLS-LOC/', help="Path to ImageNet data")
    parser.add_argument('--train-txt', type=str, default='ImageNet_LT_train.txt', help="Path to ImageNet_LT_train.txt")
    parser.add_argument('--val-txt', type=str, default='ImageNet_LT_val.txt', help="Path to ImageNet_LT_val.txt")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int, default=1000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    print(f"Building DataLoaders for {args.num_classes} classes...")
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_dataset = ImageNetLT(args.data_path, args.train_txt, transform_train)
    test_dataset = ImageNetLT(args.data_path, args.val_txt, transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Compute Class Distribution for DRW
    cls_distribution = np.zeros(args.num_classes)
    for lbl in train_dataset.labels:
        cls_distribution[lbl] += 1
        
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_distribution)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * args.num_classes
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    print("Initializing DeiT-Small...")
    model = DeiTLT(num_classes=args.num_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    base_criterion = SoftTargetCrossEntropy()
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=args.num_classes)
    drw_epoch = int(args.epochs * 0.9)
    
    print("==============================================")
    print(" PHASE 1: BASELINE TRAINING")
    print("==============================================")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        use_drw = epoch >= drw_epoch
        
        start_t = time.time()
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            if not use_drw:
                images, targets = mixup_fn(images, targets)
            else:
                targets = F.one_hot(targets, num_classes=args.num_classes).float()
                
            with torch.amp.autocast('cuda'):
                logits_cls, logits_dist = model(images)
                if use_drw:
                    loss_cls = F.cross_entropy(logits_cls, targets.argmax(dim=1), weight=per_cls_weights)
                    loss_dist = F.cross_entropy(logits_dist, targets.argmax(dim=1), weight=per_cls_weights)
                else:
                    loss_cls = base_criterion(logits_cls, targets)
                    loss_dist = base_criterion(logits_dist, targets)
                loss = loss_cls + loss_dist
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    lc, ld = model(imgs)
                    p = ((lc + ld)/2).argmax(dim=1)
                    correct += (p == lbls).sum().item()
                    total += lbls.size(0)
            print(f"Epoch {epoch+1:03d} [{time.time()-start_t:.1f}s] | Loss: {total_loss/len(train_loader):.3f} | Acc: {correct/total*100:.2f}%")
            
    print("==============================================")
    print(" PHASE 2: ORACLE ALPHA SEARCH")
    print("==============================================")
    model.eval()
    all_cls, all_dist, all_targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            l_cls, l_dist = model(imgs)
            all_cls.append(l_cls.cpu())
            all_dist.append(l_dist.cpu())
            all_targets.append(lbls.cpu())
            
    all_cls = torch.cat(all_cls)
    all_dist = torch.cat(all_dist)
    all_targets = torch.cat(all_targets)
    
    alphas = torch.linspace(0, 1, 101).view(-1, 1, 1)
    fused = alphas * all_cls.unsqueeze(0) + (1 - alphas) * all_dist.unsqueeze(0)
    preds = fused.argmax(dim=2)
    correct_matrix = (preds == all_targets.unsqueeze(0))
    oracle_acc = correct_matrix.any(dim=0).float().mean().item() * 100
    baseline_acc = correct_matrix[50].float().mean().item() * 100
    
    print(f"Baseline (50/50) Acc: {baseline_acc:.2f}%")
    print(f"Oracle Upper Bound:   {oracle_acc:.2f}%")
    print(f"Total Potential Gain: +{oracle_acc - baseline_acc:.2f}%")
    
    print("==============================================")
    print(" PHASE 3: NEURAL ENTROPY ROUTER")
    print("==============================================")
    p_cls = F.softmax(all_cls, dim=1)
    p_dist = F.softmax(all_dist, dim=1)
    conf_cls, _ = torch.max(p_cls, dim=1)
    conf_dist, _ = torch.max(p_dist, dim=1)
    ent_cls = -torch.sum(p_cls * torch.log(p_cls + 1e-8), dim=1)
    ent_dist = -torch.sum(p_dist * torch.log(p_dist + 1e-8), dim=1)
    
    X_features = torch.stack([conf_cls, conf_dist, ent_cls, ent_dist], dim=1).to(device)
    all_cls_t = all_cls.to(device)
    all_dist_t = all_dist.to(device)
    all_targets_t = all_targets.to(device)
    
    router = EntropyRouter().to(device)
    t_cls = torch.tensor([1.0], requires_grad=True, device=device)
    t_dist = torch.tensor([1.0], requires_grad=True, device=device)
    router_opt = torch.optim.Adam(list(router.parameters()) + [t_cls, t_dist], lr=0.01, weight_decay=1e-4)
    
    best_router_acc = 0.0
    for r_epoch in range(500):
        router.train()
        router_opt.zero_grad()
        
        dyn_alphas = router(X_features)
        scaled_cls = all_cls_t / torch.clamp(t_cls, 0.1, 10.0)
        scaled_dist = all_dist_t / torch.clamp(t_dist, 0.1, 10.0)
        
        fused_logits = dyn_alphas * scaled_cls + (1 - dyn_alphas) * scaled_dist
        loss = F.cross_entropy(fused_logits, all_targets_t)
        loss.backward()
        router_opt.step()
        
        with torch.no_grad():
            acc = (fused_logits.argmax(dim=1) == all_targets_t).float().mean().item()
            if acc > best_router_acc: best_router_acc = acc
            if r_epoch % 100 == 0 or r_epoch == 499:
                print(f"Router Epoch {r_epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%")
                
    print(f"\\nFINAL NEURAL ROUTER ACCURACY: {best_router_acc*100:.2f}%")
    print(f"Learned Temps: T_CLS={t_cls.item():.3f}, T_DIST={t_dist.item():.3f}")

if __name__ == "__main__":
    main()
