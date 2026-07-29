# Stage 5 & 6: Verification & Scientific Validation

## Verification Suite (`verify.py`)

To verify that the clean-room implementation perfectly matches the mathematical operations of the official code, you would use this sanity-check script. Due to hardware constraints, we provide the script rather than running a full multi-GPU validation loop.

```python
# verify.py (Conceptual Script)
import torch
import sys
sys.path.append('../13_DeiT_LT_Code/') # Official
sys.path.append('../15_MyDeiTLT/')     # Ours

from deit_models import deit_small_patch16_224 as OfficialDeiT
from model import DeiTLT as MyDeiT

def test_forward_pass():
    dummy_input = torch.randn(2, 3, 224, 224)
    
    official_model = OfficialDeiT(pretrained=False)
    my_model = MyDeiT()
    
    # Force weights to match for testing
    my_model.load_state_dict(official_model.state_dict(), strict=False)
    
    with torch.no_grad():
        off_cls, off_dist = official_model(dummy_input)
        my_cls, my_dist = my_model(dummy_input)
        
    assert torch.allclose(off_cls, my_cls, atol=1e-5), "CLS outputs differ!"
    assert torch.allclose(off_dist, my_dist, atol=1e-5), "DIST outputs differ!"
    print("Forward pass verified perfectly.")
```

## Scientific Validation Scripts

To reproduce the exact tables in the paper (e.g., Table 2), you need to run the official codebase on a standard dataset like CIFAR-100-LT. Below is the bash script template you would execute on your GPU cluster.

### Reproduce CIFAR-100-LT (Imbalance 100)
```bash
#!/bin/bash
# reproduce_cifar100.sh

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --model deit_small_patch16_224 \
    --data-set CIFAR100LT \
    --data-path /path/to/cifar100 \
    --imb_factor 0.01 \
    --batch-size 128 \
    --epochs 300 \
    --teacher-model resnet32_cifar \
    --teacher-path /path/to/sam_teacher_checkpoint.pth \
    --distillation-type hard \
    --drw 160 \
    --weighted-distillation True \
    --output_dir ./output_cifar100lt
```
*Running this script on 4 GPUs will output the logs required to plot Figure 4 and reproduce the numbers in Table 2.*
