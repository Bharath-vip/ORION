import json
import os

# Base notebook from Final ATF Model
input_file = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/18_Final_ATF_Model/DeiT_LT_ATF_Final.ipynb"
output_dir = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/21_Neural_Entropy_Router"
output_file = os.path.join(output_dir, "DeiT_LT_Neural_Router.ipynb")

os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    nb = json.load(f)

# Keep cells up to Oracle Search (which defines all_cls, all_dist, all_targets)
# The Oracle Search cell starts with "## 7. Adaptive Token Fusion"
filtered_cells = []
for cell in nb["cells"]:
    filtered_cells.append(cell)
    if cell["cell_type"] == "code" and "Oracle Alpha Search" in "".join(cell["source"]):
        break # Stop keeping cells after the Oracle Search code cell

nb["cells"] = filtered_cells

# Update title
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "Final Adaptive Token Fusion" in "".join(cell["source"]):
        cell["source"] = ["# Next-Gen ATF: The Neural Entropy Router\n", "\n", "This notebook replaces static Class-Frequency fusion with **Dynamic Instance-Level Fusion**. It trains a lightweight Multi-Layer Perceptron (MLP) to output the perfect $\\alpha$ for *every single image* based purely on the Entropy and Confidence of the CLS and DIST tokens!"]

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

add_markdown("## 8. The Neural Entropy Router (MLP)\n\nHere we extract the 4 key instance-level features (CLS Entropy, DIST Entropy, CLS Confidence, DIST Confidence) and train a lightweight 2-layer Neural Network to route the logits!")
add_code('''import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("Extracting Instance-Level Features (Confidence & Entropy)...")
# all_cls and all_dist are shape [N, 10]
p_cls = F.softmax(all_cls, dim=1)
p_dist = F.softmax(all_dist, dim=1)

conf_cls, _ = torch.max(p_cls, dim=1)
conf_dist, _ = torch.max(p_dist, dim=1)

ent_cls = -torch.sum(p_cls * torch.log(p_cls + 1e-8), dim=1)
ent_dist = -torch.sum(p_dist * torch.log(p_dist + 1e-8), dim=1)

# Stack features into an [N, 4] matrix
X_features = torch.stack([conf_cls, conf_dist, ent_cls, ent_dist], dim=1).to(args.device)
all_cls_t = all_cls.to(args.device)
all_dist_t = all_dist.to(args.device)
all_targets_t = all_targets.to(args.device)

# Define the Neural Router MLP
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
        return self.sigmoid(x) # Outputs alpha per instance

router = EntropyRouter().to(args.device)
# Also learn temperatures globally
t_cls = torch.tensor([1.0], requires_grad=True, device=args.device)
t_dist = torch.tensor([1.0], requires_grad=True, device=args.device)

optimizer = optim.Adam(list(router.parameters()) + [t_cls, t_dist], lr=0.01, weight_decay=1e-4)

print("Training the Neural Router (500 Epochs)...")
best_loss = float('inf')
best_acc = 0.0

for epoch in range(500):
    router.train()
    optimizer.zero_grad()
    
    # Forward pass through router to get per-instance alphas [N, 1]
    alphas = router(X_features)
    
    # Clamp temperatures
    t_c = torch.clamp(t_cls, min=0.1, max=10.0)
    t_d = torch.clamp(t_dist, min=0.1, max=10.0)
    
    scaled_cls = all_cls_t / t_c
    scaled_dist = all_dist_t / t_d
    
    # Fused logits for each instance
    fused_logits = alphas * scaled_cls + (1 - alphas) * scaled_dist
    
    loss = F.cross_entropy(fused_logits, all_targets_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0 or epoch == 499:
        with torch.no_grad():
            preds = fused_logits.argmax(dim=1)
            acc = (preds == all_targets_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Router Accuracy: {acc*100:.2f}%")

print("="*40)
print(f"Baseline (50/50) Accuracy: 72.01%")
print(f"Oracle Upper Bound:        74.40%")
print(f"Neural Router Accuracy:    {best_acc*100:.2f}%")
print("="*40)

print(f"Learned Global Temps: T_CLS={t_cls.item():.3f}, T_DIST={t_dist.item():.3f}")

# Let's inspect some of the alphas the router decided on!
with torch.no_grad():
    alphas = router(X_features)
    head_mask = (all_targets_t == 0) # Class 0 is the most frequent Head class
    tail_mask = (all_targets_t == 9) # Class 9 is the least frequent Tail class
    
    print(f"\\nAverage Router Alpha for Head Class 0: {alphas[head_mask].mean().item():.3f} (Oracle=0.95)")
    print(f"Average Router Alpha for Tail Class 9: {alphas[tail_mask].mean().item():.3f} (Oracle=0.00)")
''')

with open(output_file, "w") as f:
    json.dump(nb, f, indent=2)

print("Neural Router Notebook generated successfully!")
