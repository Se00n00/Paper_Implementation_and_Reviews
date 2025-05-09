import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import SinusoidalEmbeddingLayer, FeedForward, Multi_Head_Attention

# Here also, for ease of training, we would use Pre-Normallization instead of Post-Normalization
class Decoder_Layer(nn.Module):
    def __init__(self, config):
        super(Decoder_Layer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.layer_norm3 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.masked_multi_head_attention = Multi_Head_Attention(config.embed_dim, config.n_heads)
        self.multi_head_attention = Multi_Head_Attention(config.embed_dim, config.n_heads)
        self.feed_forward = FeedForward(config.embed_dim, config.ff_dim, config.dropout)
    
    def forward(self, x, encoder_output):
        x = self.layer_norm1(x)                                         # Apply layer1 normalization
        attention_output, attention_scores = self.masked_multi_head_attention(x, x, x, mask=True)
        x = x + attention_output                                        # Residual connection
        x = self.layer_norm2(x)                                         # Apply layer2 normalization
        attention_output, attention_scores = self.multi_head_attention(x, encoder_output, encoder_output, mask=False)
        x = x + attention_output                                        # Residual connection
        x = self.layer_norm3(x)                                         # Apply layer3 normalization
        x = x + self.feed_forward(x)                                    # Residual connection
        return x, attention_scores                                      # Return the output and attention scores

# Incorrectly Implemented as we itertevely require the encoder_output
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([Decoder_Layer(config) for _ in range(config.num_decoder_layers)])
            
    def forward(self, x, encoder_output):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)
        
        return x