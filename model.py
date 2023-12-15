from torch import nn
import torch
from typing import Tuple, List, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from oscar import Oscar

import numpy as np


class Transformer(nn.Module):
    def __init__(self, dictionary: List[str], max_seq_len : int=512, d_model : int = 768, num_heads : int = 12, num_layers : int = 6, d_ffn : int = 0, dropout : float = 0.1, start_token_id:int = 1, end_token_id:int=2) -> None:
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
        

    def forward(self, input : torch.Tensor, label : torch.Tensor, attn_mask : torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input)
        decoder_output = self.decoder(encoder_output, label, attn_mask)
        output = self.linear(decoder_output)
        return output
    
    def get_dictionary(self) -> List[str]:
        return self.dictionary
    
    def get_d_model(self) -> int:
        return self.d_model
    
    def get_num_heads(self) -> int:
        return self.num_heads
    
    def get_num_layers(self) -> int:
        return self.num_layers
    
    def get_max_seq_len(self) -> int:
        return self.max_seq_len
    
    def get_start_token(self) -> int:
        return self.start_token
    
    def get_end_token(self) -> int:
        return self.end_token
    
    def get_d_ffn(self) -> int:
        return self.d_ffn
    
    def get_dropout(self) -> float:
        return self.dropout   


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=512, max_seq_len : int=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute positional encodings
        even_i=torch.arange(0, self.d_model, 2).float()
        denominator=torch.pow(10000, even_i/self.d_model)
        position= torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_position_encoding=torch.sin(position/denominator)
        odd_position_encoding=torch.cos(position/denominator)
        stacked_position=torch.stack([even_position_encoding, odd_position_encoding], dim=2) 
        self.position_encoding=torch.flatten(stacked_position, start_dim=1, end_dim=2).to(device)
        
        
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, int):
            x = torch.tensor([x])
        
        embed_x = self.embedding(x)

        encoded_x = embed_x + self.position_encoding  # In-place addition

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
        x = self.multihead_attention_layer(x, x, x, None) 
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
    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, attn_mask : Any) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.multihead_attention(query, key, value, attn_mask)
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

    def forward(self, encoder_output : torch.Tensor, label : torch.Tensor, attn_mask : torch.Tensor) -> torch.Tensor:
        label = self.positional_encoding(label)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(encoder_output, label, attn_mask)
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

    def forward(self, encoder_output : torch.Tensor, decoder_input : torch.Tensor, attn_mask : torch.Tensor) -> torch.Tensor:
    
        x = self.mask_multihead_attention_layer(decoder_input, decoder_input, decoder_input, attn_mask)
        x = self.multihead_attention_layer(encoder_output, encoder_output, x, None)
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
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Set device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    indices = list(range(0, len(oscar), 10))  # This will take every second element

    # Create the subset
    subset = torch.utils.data.Subset(oscar, indices)

    # Define DataLoader for Oscar dataset
    batch_size = 64  # Change this to whatever fits in our GPU
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Training loop
    num_epochs = 5 # Change this too if we want to train for more epochs
    best_loss = float('inf')
    patience, trials = 10, 0
    accumulation_steps = 4  # Change this to the number of batches to accumulate gradients

    model.train()  # Ensure the model is in training mode
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()  # Reset gradients at the beginning of each epoch
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)

            attn_mask = torch.full([model.max_seq_len, batch_size], -np.inf)
            attn_mask = torch.triu(attn_mask, diagonal=1).to(device)

            # Forward pass
            outputs = model(inputs, targets, attn_mask)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss = loss / accumulation_steps  # Normalize the loss because it's accumulated for multiple batches
            loss.backward()  # Backward pass

            # Perform optimization step only after accumulating gradients for accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

        # Early stopping
        if avg_loss < best_loss:
            trials = 0
            best_loss = avg_loss
            torch.save(model.state_dict(), "camembert2.pth")
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
