import torch
import torch.nn as nn
import torch.nn.functional as F

# TransformerConfig
#   num_encoder_layers
#   embed_din
#   device
#   dropout_prob
#   EncodingConfig: max_len
#   embed_dim, num_heads , mask=False

class Transformer(nn.Module):
    def __init__(self, TransformerConfig):
        super(Transformer, self).__init__()
        self.embeddings = SineCosineEncoding(
            TransformerConfig.embed_dim, 
            TransformerConfig.EncodingConfig.max_len,
            TransformerConfig.device,
            TransformerConfig.dropout_prob
        )

class SineCosineEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, device='cpu', dropout_prob=0.1):
        super(SineCosineEncoding, self).__init__()
        self.embed_dim = embed_dim

        position_encoding = torch.zeros(max_len, embed_dim, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('position_encoding', position_encoding.unsqueeze(0))  # [1, max_len, embed_dim]
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads , mask=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads To Ensure Proper Concatenation"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.model_dim = embed_dim // num_heads

        self.mask = mask

        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.WO = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # Linear projections
        Q = self.WQ(q)  # [B, T, E]
        K = self.WK(k)
        V = self.WV(v)

        # Reshape for multi-head: [B, T, E] → [B, H, T, D]
        Q = Q.view(batch_size, -1, self.num_heads, self.model_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.model_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.model_dim).transpose(1, 2)

        if self.mask:
            mask = torch.triu(torch.ones(q.size(1), k.size(1), device=q.device), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T_q, T_k]
            scores = scores.masked_fill(mask, float('-inf'))


        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.model_dim**0.5
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, D]

        # Concatenate heads: [B, H, T, D] → [B, T, E]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.WO(attn_output)
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, droupout=0.5):    # ff_dim is usally higher that model_dim
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(droupout)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.droupout(x)       # apply dropout to the output of the second linear layer to reduce overfitting
        return x