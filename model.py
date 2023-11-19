from torch import nn
import torch
from typing import Tuple

# tout ça c'est le premier paragraphe de la partie 3.1 de l'article que je vous ai envoyé sur whatsapp

#Je suis pas trop sur, surtout pour les forwards

class Encoder(nn.Module):  
    def __init__(self, embed_dim : int = 768, num_heads : int = 12 , num_layers : int = 12, dropout : float = 0.1) -> None:
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, dropout : float) -> None:
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.multihead_attention_layer = MultiHeadAttentionSubLayer(embed_dim = self.embed_dim, num_heads = self.num_heads, dropout = self.dropout)
        self.position_wise_fully_connected_feed_forward_layer = PositionWiseFullyConnectedFeedForwardSubLayer(embed_dim = self.embed_dim, dropout = self.dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        attn_output, attn_output_weights = self.multihead_attention_layer(x, x, x) #faut voir ce truc la, automatiquement copilot mais x,x,x c bizarre
        x = x + attn_output
        x = self.position_wise_fully_connected_feed_forward_layer(x)
        return x

        
class MultiHeadAttentionSubLayer(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, dropout : float) -> None:
        super(MultiHeadAttentionSubLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.multihead_attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads = self.num_heads)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)

    #Ducoup la faut voir ce que c'est query, key et value, c'est trop bien copilot autocompile mes doutes il est trop fort
    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        attn_output, attn_output_weights = self.multihead_attention(query, key, value)
        return attn_output, attn_output_weights

class PositionWiseFullyConnectedFeedForwardSubLayer(nn.Module):
    def __init__(self, embed_dim : int, dropout : float, d_ffn : int = 0) -> None:
        super(PositionWiseFullyConnectedFeedForwardSubLayer, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        if d_ffn == 0: # de manière générale, d_ffn vaut 4 * embed_dim
            self.d_ffn = 4 * self.embed_dim
        else:
            self.d_ffn = d_ffn
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.d_ffn),
            nn.GELU(),
            nn.Linear(self.d_ffn, self.embed_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        self.dropout_2 = nn.Dropout(self.dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.feed_forward(x)
        return x

if __name__ == "__main__":
    encoder = Encoder()
    print(encoder)