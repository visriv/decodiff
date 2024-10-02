import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, feature_map1, feature_map2):
        """
        :param feature_map1: Tensor of shape (B, T, C, H, W)
        :param feature_map2: Tensor of shape (B, T, C, H, W)
        :return: Fused feature map of shape (B, T, C, H, W)
        """
        # Reshape feature maps to (H*W, B, C) for attention mechanism
        B, T, C, H, W = feature_map1.size()
        feature_map1 = feature_map1.view(B, C*T, -1).permute(2, 0, 1)  # (H*W, B, C*T)
        feature_map2 = feature_map2.view(B, C*T, -1).permute(2, 0, 1)  # (H*W, B, C*T)
        
        # Project queries, keys, and values for cross-attention
        query = self.query_proj(feature_map1)  # (H*W, B, CT)
        key = self.key_proj(feature_map2)      # (H*W, B, CT)
        value = self.value_proj(feature_map2)  # (H*W, B, CT)
        
        # Apply multi-head attention
        attn_output, _ = self.attn(query, key, value)
        
        # Reshape the attention output back to (B, T, C, H, W)
        fused_output = attn_output.permute(1, 2, 0).view(B, T, C, H, W)
        
        return fused_output

# Example usage
B, T, C, H, W = 4, 10, 64, 32, 32  # Example dimensions for the feature maps
feature_map1 = torch.rand(B, T, C, H, W)
feature_map2 = torch.rand(B, T, C, H, W)

fusion_module = CrossAttentionFusion(embed_dim=C*T, num_heads=8)
fused_output = fusion_module(feature_map1, feature_map2)

print(fused_output.shape)  # Should output: torch.Size([B, T, C, H, W])
