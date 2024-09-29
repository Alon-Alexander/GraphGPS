import torch
import torch.nn as nn

class CustomAttention(nn.Module):
    def __init__(self, 
                 dim_h: int, 
                 num_heads: int,
                 attn_dropout=0.0
                 ):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dim_h, num_heads, dropout=attn_dropout, batch_first=True)
    
    def forward(self, x, attn_mask, key_padding_mask):
        return self.attn.forward(x, x, x, 
                                 attn_mask=attn_mask, 
                                 key_padding_mask=key_padding_mask, 
                                 need_weights=False,
                                 )[0]
