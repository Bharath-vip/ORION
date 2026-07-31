import json
import os

input_file = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/16_Kaggle_Reproduction/DeiT_LT_Kaggle_IF50.ipynb"
output_dir = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/22_ImageNet_LT_Scaling"
output_file = os.path.join(output_dir, "DeiT_LT_ImageNet_Baseline.ipynb")

os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    nb = json.load(f)

# Update the first markdown cell
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "DeiT-LT Exact Reproduction" in "".join(cell["source"]):
        cell["source"] = ["# DeiT-LT Scaling: ImageNet-LT\n", "\n", "This notebook scales the baseline DeiT-LT model to the massive **ImageNet-LT** dataset (1000 classes). We will use a larger backbone (`DeiT-Small`) and replace the CIFAR-10 dataloaders with standard `torchvision` ImageNet-LT configurations."]

# Find and replace the dataset code block
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code_str = "".join(cell["source"])
        if "get_cifar10_lt" in code_str:
            # Replace CIFAR code with ImageNet-LT code stub
            cell["source"] = [
                "import os\n",
                "import torch\n",
                "from torchvision import datasets, transforms\n",
                "from torch.utils.data import DataLoader, Dataset\n",
                "from PIL import Image\n",
                "\n",
                "print('Downloading / Preparing ImageNet-LT splits...')\n",
                "# Kaggle users typically mount ImageNet via Kaggle Datasets.\n",
                "# Assuming ImageNet is mounted at /kaggle/input/imagenet/ILSVRC/Data/CLS-LOC/\n",
                "IMAGENET_PATH = '/kaggle/input/imagenet/ILSVRC/Data/CLS-LOC/'\n",
                "\n",
                "# You will need the ImageNet_LT_train.txt and ImageNet_LT_val.txt split files.\n",
                "# For the purpose of this scaling notebook, we provide the DataLoader structure.\n",
                "\n",
                "class ImageNetLT(Dataset):\n",
                "    def __init__(self, root, txt_file, transform=None):\n",
                "        self.img_path = []\n",
                "        self.labels = []\n",
                "        self.transform = transform\n",
                "        \n",
                "        if os.path.exists(txt_file):\n",
                "            with open(txt_file) as f:\n",
                "                for line in f:\n",
                "                    self.img_path.append(os.path.join(root, line.split()[0]))\n",
                "                    self.labels.append(int(line.split()[1]))\n",
                "        else:\n",
                "            print(f'WARNING: Split file {txt_file} not found. Please upload it to Kaggle.')\n",
                "            \n",
                "    def __len__(self):\n",
                "        return len(self.labels)\n",
                "        \n",
                "    def __getitem__(self, index):\n",
                "        path = self.img_path[index]\n",
                "        label = self.labels[index]\n",
                "        \n",
                "        try:\n",
                "            with open(path, 'rb') as f:\n",
                "                sample = Image.open(f).convert('RGB')\n",
                "        except:\n",
                "            # Dummy fallback if image is missing\n",
                "            sample = Image.new('RGB', (224, 224))\n",
                "            \n",
                "        if self.transform is not None:\n",
                "            sample = self.transform(sample)\n",
                "            \n",
                "        return sample, label\n",
                "\n",
                "# Standard ImageNet transforms\n",
                "train_transform = transforms.Compose([\n",
                "    transforms.RandomResizedCrop(224),\n",
                "    transforms.RandomHorizontalFlip(),\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
                "])\n",
                "\n",
                "val_transform = transforms.Compose([\n",
                "    transforms.Resize(256),\n",
                "    transforms.CenterCrop(224),\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
                "])\n",
                "\n",
                "print('NOTE: You must upload the ImageNet-LT txt splits to Kaggle to run this notebook.')\n",
                "# train_dataset = ImageNetLT(IMAGENET_PATH, 'ImageNet_LT_train.txt', transform=train_transform)\n",
                "# test_dataset = ImageNetLT(IMAGENET_PATH, 'ImageNet_LT_test.txt', transform=val_transform)\n",
                "\n",
                "args.num_classes = 1000\n",
                "args.epochs = 90  # ImageNet typically trains for 90 epochs, not 300\n"
            ]
            
        elif "deit_tiny_patch16_224" in code_str:
            # Upgrade model to small
            cell["source"] = [
                "# Instantiate DeiT-Small (scaled up for ImageNet 1000 classes)\n",
                "model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=args.num_classes)\n",
                "\n",
                "# Add the DIST token for DeiT-LT\n",
                "dist_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))\n",
                "model.dist_token = dist_token\n",
                "\n",
                "old_forward = model.forward_features\n",
                "def new_forward_features(self, x):\n",
                "    B = x.shape[0]\n",
                "    x = self.patch_embed(x)\n",
                "    cls_tokens = self.cls_token.expand(B, -1, -1)\n",
                "    dist_tokens = self.dist_token.expand(B, -1, -1)\n",
                "    x = torch.cat((cls_tokens, dist_tokens, x), dim=1)\n",
                "    x = x + self.pos_embed\n",
                "    x = self.pos_drop(x)\n",
                "    for blk in self.blocks:\n",
                "        x = blk(x)\n",
                "    x = self.norm(x)\n",
                "    return x[:, 0], x[:, 1]  # Return both CLS and DIST tokens\n",
                "\n",
                "import types\n",
                "model.forward_features = types.MethodType(new_forward_features, model)\n",
                "\n",
                "model.head = nn.Linear(model.embed_dim, args.num_classes)\n",
                "model.head_dist = nn.Linear(model.embed_dim, args.num_classes)\n",
                "\n",
                "def new_forward(self, x):\n",
                "    cls_token, dist_token = self.forward_features(x)\n",
                "    return self.head(cls_token), self.head_dist(dist_token)\n",
                "\n",
                "model.forward = types.MethodType(new_forward, model)\n",
                "model.to(args.device)\n",
                "print(f'Initialized DeiT-Small for {args.num_classes} classes.')\n"
            ]

with open(output_file, "w") as f:
    json.dump(nb, f, indent=2)

print("ImageNet-LT Notebook generated successfully!")
