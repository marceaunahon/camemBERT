from torch import nn
import torch
from typing import Tuple, List


class Transformer(nn.Module):
    def __init__(self, dictionary: List[str], max_seq_len : int=512, d_model : int = 768, num_heads : int = 12, num_layers : int = 6, d_ffn : int = 0, dropout : float = 0.1) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        if d_ffn == 0: # de manière générale, d_ffn vaut 4 * d_model
            self.d_ffn = 4 * self.d_model
        else:
            self.d_ffn = d_ffn
        self.dropout = dropout
        self.encoder = Encoder(max_seq_len = self.max_seq_len, d_model = self.d_model, num_heads = self.num_heads, num_layers = self.num_layers, d_ffn= self.d_ffn, dropout = self.dropout)
        self.decoder = Decoder(max_seq_len = self.max_seq_len, d_model = self.d_model, num_heads = self.num_heads, num_layers = self.num_layers, d_ffn = self.d_ffn, dropout = self.dropout)
        self.linear = nn.Linear(in_features = self.d_model, out_features = len(self.dictionary)) 
        

    def forward(self, input : List[str], output_encoding : torch.Tensor) -> torch.Tensor:
        # Je suppose que la tokenisation est déjà faite (input : List[str])
        #embedded_input = torch.tensor([self.dictionary.index(word) if word in self.dictionary else -1 for word in input], dtype=torch.long)
        #encoded_input = self.positional_encoding(embedded_input)
        x = self.encoder(input)
        x = self.decoder(x, output_encoding)
        x = self.linear(x)
        x = torch.softmax(x, dim = 1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int=768, max_seq_len : int=512):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        #self.register_buffer('positional_encoding', self._get_positional_encoding())
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        embed_x = self.embedding(x)

        even_i=torch.arange(0, self.d_model, 2).float()
        denominator=torch.pow(10000, even_i/self.d_model)
        position= torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_position_encoding=torch.sin(position/denominator)
        odd_position_encoding=torch.cos(position/denominator)
        stacked_position=torch.stack([even_position_encoding, odd_position_encoding], dim=2) 
        position_encoding=torch.flatten(stacked_position, start_dim=1, end_dim=2)
        
        encoded_x = embed_x + position_encoding

        return encoded_x
    
class Encoder(nn.Module):  
    def __init__(self, max_seq_len, d_model : int, num_heads : int, num_layers : int, d_ffn : int, dropout : float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.positional_encoding = PositionalEncoding(d_model = self.d_model, max_seq_len = self.max_seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_model, self.num_heads, self.d_ffn, self.dropout) for _ in range(num_layers)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model : int, num_heads : int, d_ffn : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.multihead_attention_layer = MultiHeadAttentionSubLayer(d_model = self.d_model, num_heads = self.num_heads, dropout = self.dropout)
        self.position_wise_fully_connected_feed_forward_layer = PositionWiseFullyConnectedFeedForwardSubLayer(d_model = self.d_model, dropout = self.dropout, d_ffn = self.d_ffn)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        attn_output, attn_output_weights = self.multihead_attention_layer(x, x, x) #faut voir ce truc la, automatiquement copilot met x,x,x c bizarre
        x = attn_output
        x = self.position_wise_fully_connected_feed_forward_layer(x)
        return x
   
class MultiHeadAttentionSubLayer(nn.Module):
    def __init__(self, d_model : int, num_heads : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.multihead_attention = nn.MultiheadAttention(embed_dim = self.d_model, num_heads = self.num_heads, dropout = self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    #Ducoup la faut voir ce que c'est query, key et value, c'est trop bien copilot autocompile mes doutes il est trop fort
    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        attn_output, attn_output_weights = self.multihead_attention(query, key, value) 
        attn_output = self.layer_norm(query + attn_output)
        return attn_output, attn_output_weights

class PositionWiseFullyConnectedFeedForwardSubLayer(nn.Module):
    def __init__(self, d_model : int, dropout : float, d_ffn : int) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.d_ffn = d_ffn
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_ffn),
            nn.GELU(),
            nn.Linear(self.d_ffn, self.d_model)
        )
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x + self.feed_forward(x)) 
        x = self.dropout_layer(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, max_seq_len : int, d_model : int, num_heads : int, num_layers : int, d_ffn : int, dropout : float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.positional_encoding = PositionalEncoding(d_model = self.d_model, max_seq_len = self.max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.d_model, self.num_heads, self.d_ffn, self.dropout) for _ in range(num_layers)])

    def forward(self, encoder_ouptut : torch.Tensor, output_embedding : torch.Tensor) -> torch.Tensor:
        x = torch.Tensor()
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(encoder_ouptut, output_embedding)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model : int, num_heads : int, d_ffn : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_ffn = d_ffn
        self.mask_multihead_attention_layer = MultiHeadAttentionSubLayer(d_model = self.d_model, num_heads = self.num_heads, dropout = self.dropout) #voir la diff entre les deux attention
        self.multihead_attention_layer = MultiHeadAttentionSubLayer(d_model = self.d_model, num_heads = self.num_heads, dropout = self.dropout)
        self.position_wise_fully_connected_feed_forward_layer = PositionWiseFullyConnectedFeedForwardSubLayer(d_model = self.d_model, d_ffn = d_ffn, dropout = self.dropout)

    def forward(self, encoder_output : torch.Tensor, output_embedding : torch.Tensor) -> torch.Tensor:
        attn_output, attn_output_weights = self.mask_multihead_attention_layer(output_embedding, output_embedding, output_embedding)
        x = attn_output
        attn2_output, attn2_output_weights = self.multihead_attention_layer(encoder_output, encoder_output, x)
        x = attn2_output
        x = self.position_wise_fully_connected_feed_forward_layer(x)
        return x
    
if __name__ == "__main__":
    dictionnary = ["a", "b", "c"]
    model = Transformer(dictionnary)
    print(model)