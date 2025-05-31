import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel


class ViTconfig:
    model_type = "ViT"
    patch_size = 16
    image_size = 224
    patch_length = ((image_size - patch_size) // patch_size + 1) ** 2
    in_channels = 3
    embed_dim = 768
    num_layers = 12
    num_heads = 12
    num_classes = 100

# ViT (Vision Transformer) Model to be used with HuggingFace Transformers
class ViTTransformerForClassification(PreTrainedModel):
    config_class = ViTconfig
    def __init__(self, config):
        super(ViTTransformerForClassification, self).__init__(config)
        self.vit = ViT(config)
        self.linear = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, x):
        x = self.vit(x)
        return self.linear(x[:, 0])

# Reference : [ ViTAX: Building a Vision Transformer from Scratch ]
# https://maurocomi.com/blog/vit.html
class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim
        )
        self.postional_encoding = PositionalEncoding(
            embed_dim=config.embed_dim,
            patch_length = config.patch_length
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))  # Class token for classification tasks
        
        self.layers = nn.ModuleList(
            [ViT_layer(config.embed_dim, config.num_heads) for _ in range(config.num_layers)]
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)  # (B, L+1, D)
        x = self.postional_encoding(x)

        for layer in self.layers:
            x = layer(x)
        return x


class ViT_layer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ViT_layer, self).__init__()

        self.attention = ViTMulti_Head_Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.feed_forward = ViTFeedForward(embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        norm1 = self.norm1(x)  # Layer Normalization before Attention
        attn_output = self.attention(norm1, norm1, norm1)
        norm2 = self.norm2(norm1 + attn_output)

        ff_output = self.feed_forward(norm2)
        x = norm2 + ff_output

        x = self.norm3(x)  # Layer Normalization after Feed Forward
        x = self.dropout(x)
        return x


# Reference : [VIT part 1: Patchify Images using PyTorch Unfold]
# https://mrinath.medium.com/vit-part-1-patchify-images-using-pytorch-unfold-716cd4fd4ef6
class PatchEmbedding(nn.Module):
    """Embeddeds the Patches of images i.e 
        Batch of Image [B, C, H, W] -> Batch of Square Image Patches [B, P*P*C, num_patches_H*num_patches_W]
        
            for [B, 3, 24, 24] with patch_size=3, this will be converted to
            (24-3)3 +1 ) = 8:> 8 X 8 = 64 patches with each patch of size 3x3
        
        [B, P*P*C, num_patches_H*num_patches_W] -> reshaped [B, L, D]
        where P is the patch size and D is the embedding dimension.

    Here nn.Unfold is used to extract patches (local Blocks) from the input image.
    """

    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size = patch_size, stride=patch_size)
        self.linear = nn.Linear(patch_size * patch_size * in_channels, embed_dim)


    def forward(self, x):
        x = self.unfold(x)      # [B, C*P*P, L]
        x = x.transpose(1, 2)   # [B, L, C*P*P]
        x = self.linear(x)      # [B, L, D]
        return x

class PositionalEncoding(nn.Module):
    """Adds Positional Information to the input Embeddings
        patch_length: Length of the Patch Embedding is similiar to the max_length in NLP.
    """

    def __init__(self, embed_dim, patch_length):
        super(PositionalEncoding, self).__init__()

        self.pos_embedding = nn.Parameter(torch.zeros(1, patch_length+1, embed_dim))    # Here i have added patch_length+1 to account for the class token.
    
    def forward(self, x):
        x = x + self.pos_embedding
        return x

class ViTFeedForward(nn.Module):
    """Feed Forward Network for ViT Layer
        This is a simple MLP with GELU activation and Layer Normalization.
    """
    def __init__(self, embed_dim):
        super(ViTFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x

class ViTMulti_Head_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ViTMulti_Head_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.WO = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, q, k, v):
        B = q.size(0)
        Q_size = q.size(1)
        K_size = k.size(1)
        V_size = v.size(1)
        assert K_size == V_size, "Key and Value must have the same size"

        Q = self.WQ(q)
        K = self.WK(k)
        V = self.WV(v)

        Q = Q.view(B, Q_size, self.num_heads, self.head_dim).transpose(1, 2)    # [B, Q_size, Embed_dim] > [B, Q_size, H, D] > [B, H, Q_size, D]
        K = K.view(B, K_size, self.num_heads, self.head_dim).transpose(1, 2)    # [B, K_size, Embed_dim] > [B, K_size, H, D] > [B, H, K_size, D]
        V = V.view(B, V_size, self.num_heads, self.head_dim).transpose(1, 2)    # [B, V_size, Embed_dim] > [B, V_size, H, D] > [B, H, V_size, D]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_scores = F.softmax(scores, dim=-1)    # Output Shape: [B, H, Q_size, K_size]
        attention_output = torch.matmul(attention_scores, V)    # Output Shape: [B, H, V_size, D]

        attention_output = attention_output.transpose(1, 2).contiguous()    # [B, Q_size, H, D]
        attention_output = attention_output.view(B, Q_size, self.embed_dim)    # [B, Q_size, Embed_dim]
        output = self.WO(attention_output)    # [B, Q_size, Embed_dim]
        return output