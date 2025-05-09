import torch
import torch.nn as nn
import torch.nn.functional as F

# Adding positional Information using sinusoidal function
class SinusoidalEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, device):
        super(SinusoidalEmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        self.register_buffer("positional_embedding", self._get_positional_encoding(max_length, embed_size, device))
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
    
    def _get_positional_encoding(self, max_length, embed_size, device):
        pe = torch.zeros(max_length, embed_size, device=device)                              # Create a tensor of zeros of size (max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)              # Create a tensor of size (max_length, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))    # Create a tensor of exp values of 0 to embed_size/2
        
        pe[:, 0::2] = torch.sin(position * div_term)                                          # Apply sin function to even indices, start=0 , step=2
        pe[:, 1::2] = torch.cos(position * div_term)                                          # Apply cos function to odd indices, start=1, step=2
        pe = pe.unsqueeze(0)                                                                  # shape: (1, max_length, embed_size)
        return pe

    def forward(self, x):
        word_embedding = self.embedding(x)                                                  # Convert unique word tokens to word embeddings
        
        positional_embeddings = self.positional_embedding[:, :x.size(-2), :].to(x.device)   # Get sinosudal indicies information as positional embeddings          Shape: (1, Seqlen, embed_size)
        x = word_embedding + positional_embeddings                                          # Adds word embedding to positional embedding
        x = self.layer_norm(x)                                                              # Apply layer normalization
        return x

class Multi_Head_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Multi_Head_Attention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.WO = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, q, k, v, masked=False):
        batch_size = q.size(0)
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)
        
        # Linear projections
        Q = self.WQ(q)  # [B, L_q, E]
        K = self.WK(k)  # [B, L_k, E]
        V = self.WV(v)  # [B, L_v, E]
        
        # Reshape for multi-head: [B, T, E] → [B, H, L, D]
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # Scaled dot-product attention [B, H, L, L]
        
        # Optional Assertion
        assert q_len == k_len, "Query and Key lengths must be equal for self-attention"

        if masked:
            mask = torch.triu(torch.ones(q_len, k_len, device=q.device), diagonal=1).bool() # Create causal mask (for decoder self-attention)
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, L_q, L_k]
            scores = scores.masked_fill(mask, float('-inf'))
        
        attention_scores = F.softmax(scores, dim=-1)  # Attention Scores[B, H, T_q, T_k]
        attention_output = torch.matmul(attention_scores, V)  # [B, H, T_q, D]

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)# Concatenate heads: [B, H, T, D] → [B, T, E]
        output = self.WO(attention_output)
        
        return output, attention_scores

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