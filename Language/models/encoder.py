import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import SinusoidalEmbeddingLayer, FeedForward, Multi_Head_Attention

# Though this Implementation is based on the paper "Attention is All You Need" by Vaswani et al. (2017) but
# for ease of training, we would use Pre-Normallization instead of Post-Normalization
class Encoder_Layer(nn.Module):
    def __init__(self, config):
        super(Encoder_Layer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.multi_head_attention = Multi_Head_Attention(config.embed_dim, config.n_heads)
        self.feed_forward = FeedForward(config.embed_dim, config.ff_dim, config.dropout)
    
    def forward(self, x, mask=False):
        x = self.layer_norm1(x)                                         # Apply layer1 normalization
        attention_output, attention_scores = self.multi_head_attention(x, x, x, mask)
        x = x + attention_output                                        # Residual connection
        x = self.layer_norm2(x)                                         # Apply layer2 normalization
        x = x + self.feed_forward(x)                                    # Residual connection
        return x, attention_scores                                      # Return the output and attention scores

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([Encoder_Layer(config) for _ in range(config.num_encoder_layers)])
    
    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x