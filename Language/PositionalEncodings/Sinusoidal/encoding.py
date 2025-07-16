import torch
import torch.nn as nn

import math

class SinusoidalEncodings(nn.Module):
    def __init__(self, d_model, max_len):
        super(SinusoidalEncodings, self).__init__()

        pos = torch.arange(0, max_len).unsqueeze(1).float() # Creates [MAX_LEN, 1] tensor
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float()/d_model) # Numerically Unstable
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        PE = torch.zeros(max_len, d_model)
        PE[:, 0::2] = torch.sin(pos/div_term)
        PE[:, 1::2] = torch.cos(pos/div_term)

        PE = PE.unsqueeze(0) # [MAX_LEN, d_model] > [1, MAX_LEN, d_model]
        self.register_buffer("PE", PE)


    def forward(self, embeddings): # Takes [B, Sequence Length: L] as input Embeddings
        
        positional_info = self.PE[:, :embeddings.size(1), :]
        return embeddings + positional_info