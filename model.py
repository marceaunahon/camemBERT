from torch import nn
import torch
from typing import Tuple, List
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from roberta import Roberta
from oscar import Oscar


class Transformer(nn.Module):
    def __init__(self, dictionary: List[str], max_seq_len : int=200, d_model : int = 768, num_heads : int = 12, num_layers : int = 6, d_ffn : int = 0, dropout : float = 0.1, start_token_id:int = 1, end_token_id:int=2) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.start_token = start_token_id
        self.end_token = end_token_id
        if d_ffn == 0: # de manière générale, d_ffn vaut 4 * d_model
            self.d_ffn = 4 * self.d_model
        else:
            self.d_ffn = d_ffn
        self.dropout = dropout
        self.encoder = Encoder(len(self.dictionary), max_seq_len = self.max_seq_len, d_model = self.d_model, num_heads = self.num_heads, num_layers = self.num_layers, d_ffn= self.d_ffn, dropout = self.dropout)
        self.decoder = Decoder(len(self.dictionary), max_seq_len = self.max_seq_len, d_model = self.d_model, num_heads = self.num_heads, num_layers = self.num_layers, d_ffn = self.d_ffn, dropout = self.dropout)
        self.linear = nn.Linear(in_features = self.d_model, out_features = len(self.dictionary)) 
        

    def forward(self, input : torch.Tensor, label : torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input)
        decoder_output = self.decoder(encoder_output, label)
        output = self.linear(decoder_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=768, max_seq_len : int=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Precompute positional encodings
        even_i=torch.arange(0, self.d_model, 2).float()
        denominator=torch.pow(10000, even_i/self.d_model)
        position= torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_position_encoding=torch.sin(position/denominator)
        odd_position_encoding=torch.cos(position/denominator)
        stacked_position=torch.stack([even_position_encoding, odd_position_encoding], dim=2) 
        self.position_encoding=torch.flatten(stacked_position, start_dim=1, end_dim=2)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, int):
            x = torch.tensor([x])
        
        embed_x = self.embedding(x)
        encoded_x = embed_x.add_(self.position_encoding)  # In-place addition

        return encoded_x
    
class Encoder(nn.Module):  
    def __init__(self, vocab_size:int, max_seq_len, d_model : int, num_heads : int, num_layers : int, d_ffn : int, dropout : float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.positional_encoding = PositionalEncoding(vocab_size, d_model = self.d_model, max_seq_len = self.max_seq_len)
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
        # x, _ = self.multihead_attention_layer(x, x, x) #faut voir ce truc la, automatiquement copilot met x,x,x c bizarre
        # x = self.position_wise_fully_connected_feed_forward_layer(x)
        x = self.multihead_attention_layer(x, x, x) #faut voir ce truc la, automatiquement copilot met x,x,x c bizarre
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
        x, _ = self.multihead_attention(query, key, value)
        x = self.layer_norm(query + x)
        return x

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
    def __init__(self, vocab_size, max_seq_len : int, d_model : int, num_heads : int, num_layers : int, d_ffn : int, dropout : float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.positional_encoding = PositionalEncoding(vocab_size, d_model = self.d_model, max_seq_len = self.max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.d_model, self.num_heads, self.d_ffn, self.dropout) for _ in range(num_layers)])

    def forward(self, encoder_output : torch.Tensor, label : torch.Tensor) -> torch.Tensor:
        label = self.positional_encoding(label)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(encoder_output, label)
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

    def forward(self, encoder_output, decoder_input):
    
        x = self.mask_multihead_attention_layer(decoder_input, decoder_input, decoder_input)
        x = self.multihead_attention_layer(encoder_output, encoder_output, x)
        x = self.position_wise_fully_connected_feed_forward_layer(x)
        return x

    
if __name__ == "__main__":

    # Initialize Oscar object
    oscar = Oscar(language="fr", split="train")

    # Define Roberta model
    dictionary = list(oscar.get_vocab().keys())
    # model_name = "roberta-base"  
    # model = Roberta(model_name) # si on utilise Roberta : faire un controle h et remplacer tous les models.parameters() par model.model.parameters()
    model = Transformer(dictionary) # si on utilise model : faire un controle h et remplacer tous les models.models.parameters() par model.parameters()

    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Set device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define DataLoader for Oscar dataset
    batch_size = 32  # Change this to whatever fits in our GPU
    dataloader = DataLoader(range(len(oscar)), batch_size=batch_size, shuffle=True)

    # Training loop
    num_epochs = 1 # Change this too if we want to train for more epochs
    best_loss = float('inf')
    patience, trials = 10, 0
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch_idx in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Get batch data from the Oscar dataset
            # Convert batch_idx to list of inputs
            batch_idx = batch_idx.tolist()
            inputs = [oscar.get_masked_item(i) for i in batch_idx]
            inputs = torch.tensor(inputs).to(device)
            
            # Get the correct targets
            targets = [oscar[i] for i in batch_idx]


            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

        # Early stopping
        if avg_loss < best_loss:
            trials = 0
            best_loss = avg_loss
            torch.save(model.state_dict(), "camembert_model.pth")
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break