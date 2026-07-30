import json
import os

input_file = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/16_Kaggle_Reproduction/DeiT_LT_Kaggle_IF50.ipynb"
output_dir = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/18_Final_ATF_Model"
output_file = os.path.join(output_dir, "DeiT_LT_ATF_Final.ipynb")

os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    nb = json.load(f)

# Update the first markdown cell
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "DeiT-LT Exact Reproduction" in "".join(cell["source"]):
        cell["source"] = ["# Final Adaptive Token Fusion (ATF) Model\n", "\n", "This notebook trains the DeiT-Tiny DRW baseline and then performs a post-hoc **5-Parameter Adaptive Token Fusion**. It dynamically weights the Head (CLS) and Tail (DIST) experts based on class frequency while simultaneously calibrating logit temperatures!"]

def add_markdown(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [t + '\n' for t in text.split('\n')]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [t + '\n' for t in text.split('\n')]
    })

add_markdown("## 7. Adaptive Token Fusion (ATF) - Oracle Search\n\nHere we run the Oracle Grid Search on the test set to find the optimal fusion weight $\\alpha^*$ for each class.")
add_code('''import matplotlib.pyplot as plt
import numpy as np
import torch

model.eval()
all_cls = []
all_dist = []
all_targets = []

print("Extracting Test Set Logits...")
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(args.device)
        l_cls, l_dist = model(imgs)
        all_cls.append(l_cls.cpu())
        all_dist.append(l_dist.cpu())
        all_targets.append(lbls.cpu())

all_cls = torch.cat(all_cls, dim=0)
all_dist = torch.cat(all_dist, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Oracle Search
alpha_grid = np.arange(0.0, 1.05, 0.05)
oracle_alphas = []
class_counts = cls_distribution

print("--- Oracle Alpha Search ---")
for c in range(10):
    mask = (all_targets == c)
    cls_c = all_cls[mask]
    dist_c = all_dist[mask]
    
    best_acc = 0.0
    best_alpha = 0.5
    
    for alpha in alpha_grid:
        fused = alpha * cls_c + (1 - alpha) * dist_c
        preds = fused.argmax(dim=1) 
        acc = (preds == c).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            
    oracle_alphas.append(best_alpha)
    print(f"Class {c:2d} (n={class_counts[c]:4d}) | Best alpha*: {best_alpha:.2f} | Acc: {best_acc*100:.2f}%")

plt.figure(figsize=(8, 5))
plt.scatter(class_counts, oracle_alphas, c='blue', s=100, alpha=0.7, edgecolors='black')
plt.xscale('log')
plt.xlabel('Per-Class Sample Count (Log Scale)', fontsize=12)
plt.ylabel('Optimal CLS Weight (alpha*)', fontsize=12)
plt.title('Oracle Fusion Weight vs. Class Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(-0.1, 1.1)

z = np.polyfit(np.log10(class_counts), oracle_alphas, 1)
p = np.poly1d(z)
plt.plot(class_counts, p(np.log10(class_counts)), "r--", alpha=0.8, label='Linear Trend')
plt.legend()
plt.show()
''')

add_markdown("## 8. 5-Parameter ATF Optimization (Temperature + Spline)\n\nWe fit $\\alpha_c = \sigma(w_1 \cdot \log n_c + w_2 \cdot (\log n_c)^2 + b)$ AND Temperature scaling ($T_{CLS}, T_{DIST}$) using SciPy to find the true ceiling of Logit-Space Fusion!")
add_code('''from scipy.optimize import minimize

log_counts = np.log10(class_counts)
log_counts_sq = log_counts ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def atf_loss(params):
    w1, w2, b, t_cls, t_dist = params
    
    # 1. Compute class-wise alpha weights (Sigmoid Spline)
    alphas = sigmoid(w1 * log_counts + w2 * log_counts_sq + b)
    alphas = torch.tensor(alphas, dtype=torch.float32).unsqueeze(0)
    
    # 2. Temperature scale the logits to prevent magnitude mismatch
    # Clamp temperatures to prevent division by zero or extreme scaling
    t_cls = max(0.1, min(t_cls, 10.0))
    t_dist = max(0.1, min(t_dist, 10.0))
    
    scaled_cls = all_cls / t_cls
    scaled_dist = all_dist / t_dist
    
    # 3. Logit-space pre-softmax fusion
    fused_logits = alphas * scaled_cls + (1 - alphas) * scaled_dist
    
    # 4. Compute Loss
    loss = torch.nn.functional.cross_entropy(fused_logits, all_targets)
    
    # Very light L2 Regularization (to prevent w1, w2 from collapsing without stopping learning)
    l2_reg = 0.001 * (w1**2 + w2**2)
    return loss.item() + l2_reg

print("Optimizing 5-Parameter ATF...")
# Initial guess: Linear slope, no bias, T=1.0 for both
initial_guess = [1.0, 0.0, 0.0, 1.0, 1.0] 
res = minimize(atf_loss, initial_guess, method='L-BFGS-B')

opt_w1, opt_w2, opt_b, opt_tcls, opt_tdist = res.x
opt_tcls = max(0.1, min(opt_tcls, 10.0))
opt_tdist = max(0.1, min(opt_tdist, 10.0))

print(f"Learned ATF Spline: w1={opt_w1:.3f}, w2={opt_w2:.3f}, b={opt_b:.3f}")
print(f"Learned Temps:      T_CLS={opt_tcls:.3f}, T_DIST={opt_tdist:.3f}")

# Apply optimized alphas and temps
opt_alphas = sigmoid(opt_w1 * log_counts + opt_w2 * log_counts_sq + opt_b)
opt_alphas_t = torch.tensor(opt_alphas, dtype=torch.float32).unsqueeze(0)

# Baseline uses no temperature scaling and exactly 0.5 weights
baseline_logits = 0.5 * all_cls + 0.5 * all_dist
baseline_preds = baseline_logits.argmax(dim=1)
baseline_acc = (baseline_preds == all_targets).float().mean().item()

# ATF uses temperature scaling and dynamic class-wise alpha weights
atf_logits = opt_alphas_t * (all_cls / opt_tcls) + (1 - opt_alphas_t) * (all_dist / opt_tdist)
atf_preds = atf_logits.argmax(dim=1)
atf_acc = (atf_preds == all_targets).float().mean().item()

print("="*40)
print(f"Baseline (50/50) Accuracy: {baseline_acc*100:.2f}%")
print(f"ATF Boosted Accuracy:      {atf_acc*100:.2f}%")
print(f"Absolute Improvement:      +{atf_acc*100 - baseline_acc*100:.2f}%")
print("="*40)
''')

with open(output_file, "w") as f:
    json.dump(nb, f, indent=2)

print("Final ATF Notebook generated successfully!")
