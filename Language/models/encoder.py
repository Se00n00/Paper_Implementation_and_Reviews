import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import SinusoidalEmbeddingLayer, FeedForward, Multi_Head_Attention

# This Implementation is based on the paper "Attention is All You Need" by Vaswani et al. (2017) though
# for ease of training, we would use Pre-Normallization instead of Post-Normalization
class Encoder_Layer(nn.Module):
    def __init__(self, embed_dim, n_heads, d_ff, dropout=0.1):
        super(Encoder_Layer, self).__init__()
