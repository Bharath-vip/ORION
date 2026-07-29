import torch
import torch.nn as nn
from transformer import TransformerBlock

class DeiTLT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Dual Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional Encoding (patches + 2 tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dual Classification Heads
        self.head_cls = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patchify and flatten
        x = self.patch_embed(x) # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2) # (B, 196, embed_dim)
        
        # Prepend tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1) # (B, 198, embed_dim)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Pass through Transformer
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Extract experts
        cls_expert = x[:, 0]
        dist_expert = x[:, 1]
        
        # Get logits
        logits_cls = self.head_cls(cls_expert)
        logits_dist = self.head_dist(dist_expert)
        
        return logits_cls, logits_dist
