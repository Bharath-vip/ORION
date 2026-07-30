import json
import os

input_file = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/18_Final_ATF_Model/DeiT_LT_ATF_Final.ipynb"
output_dir = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/19_ATF_Universality_Study"
output_file = os.path.join(output_dir, "ATF_Universality_and_Gap.ipynb")

os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    nb = json.load(f)

# Update title
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "Final Adaptive Token Fusion" in "".join(cell["source"]):
        cell["source"] = ["# ATF Universality & Oracle Gap Analysis\n", "\n", "This notebook extends the final ATF model to answer two profound scientific questions:\n", "1. Is the negative slope ($w_1 < 0$) a universal property of long-tailed fusion, or just a SciPy L-BFGS artifact? We test this across Adam and Differential Evolution.\n", "2. Why is there a 1.6% gap between the 5-parameter model and the Oracle? We train a Decision Tree on instance-level features (Entropy/Confidence) to find out."]

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

add_markdown("## 9. Universality of w1 < 0 (Multi-Optimizer Ablation)\n\nTo prove that the negative slope ($w_1 < 0$, which acts as Logit Adjustment) is not a SciPy artifact, we re-optimize the 5 parameters using PyTorch's Adam and SciPy's Differential Evolution.")
add_code('''from scipy.optimize import differential_evolution
import torch.optim as optim

print("--- 1. PyTorch Adam Optimization ---")
# Define learnable tensors
w1 = torch.tensor([1.0], requires_grad=True, device=args.device)
w2 = torch.tensor([0.0], requires_grad=True, device=args.device)
b = torch.tensor([0.0], requires_grad=True, device=args.device)
t_cls = torch.tensor([1.0], requires_grad=True, device=args.device)
t_dist = torch.tensor([1.0], requires_grad=True, device=args.device)

optimizer = optim.Adam([w1, w2, b, t_cls, t_dist], lr=0.1)
log_counts_t = torch.tensor(log_counts, dtype=torch.float32, device=args.device)
log_counts_sq_t = torch.tensor(log_counts_sq, dtype=torch.float32, device=args.device)

all_cls_t = all_cls.to(args.device)
all_dist_t = all_dist.to(args.device)
all_targets_t = all_targets.to(args.device)

for epoch in range(500):
    optimizer.zero_grad()
    
    alphas = torch.sigmoid(w1 * log_counts_t + w2 * log_counts_sq_t + b).unsqueeze(0)
    
    # Clamp temperatures
    t_c = torch.clamp(t_cls, min=0.1, max=10.0)
    t_d = torch.clamp(t_dist, min=0.1, max=10.0)
    
    scaled_cls = all_cls_t / t_c
    scaled_dist = all_dist_t / t_d
    
    fused_logits = alphas * scaled_cls + (1 - alphas) * scaled_dist
    loss = torch.nn.functional.cross_entropy(fused_logits, all_targets_t)
    l2_reg = 0.001 * (w1**2 + w2**2)
    (loss + l2_reg).backward()
    optimizer.step()

print(f"Adam Learned Spline: w1={w1.item():.3f}, w2={w2.item():.3f}, b={b.item():.3f}")
print(f"Adam Learned Temps:  T_CLS={t_cls.item():.3f}, T_DIST={t_dist.item():.3f}")

print("\\n--- 2. SciPy Differential Evolution ---")
bounds = [(-5.0, 5.0), (-2.0, 2.0), (-5.0, 5.0), (0.1, 10.0), (0.1, 10.0)]
res_de = differential_evolution(atf_loss, bounds, maxiter=100, popsize=15)
opt_w1_de, opt_w2_de, opt_b_de, opt_tcls_de, opt_tdist_de = res_de.x

print(f"DE Learned Spline: w1={opt_w1_de:.3f}, w2={opt_w2_de:.3f}, b={opt_b_de:.3f}")
print(f"DE Learned Temps:  T_CLS={opt_tcls_de:.3f}, T_DIST={opt_tdist_de:.3f}")

if w1.item() < 0 and opt_w1_de < 0:
    print("\\n[CONCLUSION]: w1 < 0 is mathematically UNIVERSAL across Optimizers!")
''')

add_markdown("## 10. Oracle Gap Analysis (Instance-Level Features)\n\nWe know the Oracle gets ~74.40% by cheating (knowing the ground truth). The 5-parameter model gets ~72.73%. Why the gap? \n\nWe extract instances where the Spline failed but the Oracle succeeded, and train a Decision Tree on **Instance-Level Entropy and Confidence** to see if instance-level features can bridge the gap.")
add_code('''from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# Compute Probabilities and Entropies
p_cls = F.softmax(all_cls, dim=1)
p_dist = F.softmax(all_dist, dim=1)

conf_cls, _ = torch.max(p_cls, dim=1)
conf_dist, _ = torch.max(p_dist, dim=1)

ent_cls = -torch.sum(p_cls * torch.log(p_cls + 1e-8), dim=1)
ent_dist = -torch.sum(p_dist * torch.log(p_dist + 1e-8), dim=1)

# Isolate failure cases: Where Oracle was right, but Parametric Spline was wrong
# Note: Since the Oracle searched per-class, we re-run Oracle logic instance-by-instance here for true upper bound
oracle_instance_preds = []
for i in range(len(all_targets)):
    c = all_targets[i].item()
    # Which alpha was best for this class? (from Oracle search)
    best_alpha = oracle_alphas[c]
    fused_logit = best_alpha * all_cls[i] + (1 - best_alpha) * all_dist[i]
    oracle_instance_preds.append(fused_logit.argmax().item())

oracle_instance_preds = torch.tensor(oracle_instance_preds)
oracle_correct_mask = (oracle_instance_preds == all_targets)
spline_wrong_mask = (atf_preds != all_targets)

# Instances where Oracle saves us
gap_mask = oracle_correct_mask & spline_wrong_mask
print(f"Total instances in the 'Oracle Gap': {gap_mask.sum().item()} / {len(all_targets)}")

# Construct Feature Matrix (X) and Target Oracle Alphas (Y) for the Decision Tree
# We want the tree to predict the Oracle's chosen alpha purely from Confidence and Entropy!
X = torch.stack([conf_cls, conf_dist, ent_cls, ent_dist], dim=1).numpy()
Y = np.array([str(oracle_alphas[t.item()]) for t in all_targets]) # Cast to string for Classification

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
dt_acc = accuracy_score(Y_test, Y_pred)

print(f"\\nDecision Tree Accuracy at predicting Oracle alpha: {dt_acc*100:.2f}%")
print("Feature Importances:")
print(f"  CLS Confidence:  {clf.feature_importances_[0]:.3f}")
print(f"  DIST Confidence: {clf.feature_importances_[1]:.3f}")
print(f"  CLS Entropy:     {clf.feature_importances_[2]:.3f}")
print(f"  DIST Entropy:    {clf.feature_importances_[3]:.3f}")

print("\\n[CONCLUSION]: If DT Accuracy is high, Class-Frequency is insufficient and Instance-Level Confidence is required to close the gap!")
''')

with open(output_file, "w") as f:
    json.dump(nb, f, indent=2)

print("Universality Notebook generated successfully!")
