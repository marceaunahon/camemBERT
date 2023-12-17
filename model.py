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

    def __init__(self, dictionary: List[str], max_seq_len: int = 200, d_model: int = 768, num_heads: int = 12, num_layers: int = 6, d_ffn: int = 0, dropout: float = 0.1, start_token_id: int = 1, end_token_id: int = 2) -> None:
        """
        Transformer model for natural language processing

        Args:
            dictionary (List[str]): list of all the tokens in the dictionary
            max_seq_len (int, optional): maximum length of the sequence. Defaults to 200.
            d_model (int, optional): dimension of the model. Defaults to 768.
            num_heads (int, optional): number of heads for the multi-head attention. Defaults to 12.
            num_layers (int, optional): number of layers for the encoder and the decoder. Defaults to 6.
            d_ffn (int, optional): dimension of the feed forward network. Defaults to 0.
            dropout (float, optional): dropout rate. Defaults to 0.1.
            start_token_id (int, optional): id of the start token. Defaults to 1.
            end_token_id (int, optional): id of the end token. Defaults to 2.

        Returns:
            None
        """

        super().__init__()
        #  set the parameters
        self.dictionary = dictionary
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.start_token = start_token_id
        self.end_token = end_token_id
        #  In general, d_ffn = 4 * d_model
        self.d_ffn = 4 * self.d_model if d_ffn == 0 else d_ffn
        self.dropout = dropout
        #  initialize the encoder and the decoder
        self.encoder = Encoder(len(self.dictionary), max_seq_len=self.max_seq_len, d_model=self.d_model,
                               num_heads=self.num_heads, num_layers=self.num_layers, d_ffn=self.d_ffn, dropout=self.dropout)
        self.decoder = Decoder(len(self.dictionary), max_seq_len=self.max_seq_len, d_model=self.d_model,
                               num_heads=self.num_heads, num_layers=self.num_layers, d_ffn=self.d_ffn, dropout=self.dropout)
        #  initialize the linear layer
        self.linear = nn.Linear(in_features=self.d_model,
                                out_features=len(self.dictionary))

    def forward(self, input: torch.Tensor, label: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model

        Args:
            input (torch.Tensor): input of the model
            label (torch.Tensor): target of the model
            attn_mask (torch.Tensor): attention mask of the model

        Returns:
            output (torch.Tensor): output of the model
        """
        encoder_output = self.encoder(input)
        decoder_output = self.decoder(encoder_output, label, attn_mask)
        output = self.linear(decoder_output)
        return output

    def get_dictionary(self) -> List[str]:
        "Return the dictionary"
        return self.dictionary

    def get_d_model(self) -> int:
        "Return the dimension of the model"
        return self.d_model

    def get_num_heads(self) -> int:
        "Return the number of heads"
        return self.num_heads

    def get_num_layers(self) -> int:
        "Return the number of layers"
        return self.num_layers

    def get_max_seq_len(self) -> int:
        "Return the maximum length of the sequence"
        return self.max_seq_len

    def get_start_token(self) -> int:
        "Return the id of the start token"
        return self.start_token

    def get_end_token(self) -> int:
        "Return the id of the end token"
        return self.end_token

    def get_d_ffn(self) -> int:
        "Return the dimension of the feed forward network"
        return self.d_ffn

    def get_dropout(self) -> float:
        "Return the dropout rate"
        return self.dropout


class PositionalEncoding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int = 768, max_seq_len: int = 200):
        """
        Positional encoding for the transformer model

        Args:
            vocab_size (int): size of the vocabulary
            d_model (int, optional): dimension of the model. Defaults to 768.
            max_seq_len (int, optional): maximum length of the sequence. Defaults to 200.

        Returns:
            None
        """
        super().__init__()

        #  set the parameters
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Initialize the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        #  set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute positional encodings
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_position_encoding = torch.sin(position/denominator)
        odd_position_encoding = torch.cos(position/denominator)
        stacked_position = torch.stack(
            [even_position_encoding, odd_position_encoding], dim=2)
        self.position_encoding = torch.flatten(
            stacked_position, start_dim=1, end_dim=2).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model

        Args:
            x (torch.Tensor): input of the model

        Returns:
            encoded_x (torch.Tensor): encoded input of the model
        """

        #  check if x is a list or a tensor
        if isinstance(x, int):
            x = torch.tensor([x])

        #  get the embedding of the input
        embed_x = self.embedding(x)

        #  add the positional encoding to the embedding
        encoded_x = embed_x + self.position_encoding

        return encoded_x


class Encoder(nn.Module):

    def __init__(self, vocab_size: int, max_seq_len, d_model: int, num_heads: int, num_layers: int, d_ffn: int, dropout: float) -> None:
        """
        Encoder model for the transformer model

        Args:
            vocab_size (int): size of the vocabulary
            max_seq_len (int): maximum length of the sequence
            d_model (int): dimension of the model
            num_heads (int): number of heads for the multi-head attention
            num_layers (int): number of layers for the encoder and the decoder
            d_ffn (int): dimension of the feed forward network
            dropout (float): dropout rate

        Returns:
            None
        """
        super().__init__()

        #  set the parameters
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.dropout = dropout

        #  initialize the positional encoding and the encoder layers
        self.positional_encoding = PositionalEncoding(
            vocab_size, d_model=self.d_model, max_seq_len=self.max_seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(
            self.d_model, self.num_heads, self.d_ffn, self.dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model

        Args:
            x (torch.Tensor): input of the model

        Returns:
            x (torch.Tensor): output of the model
        """
        #  add the positional encoding to the input
        x = self.positional_encoding(x)

        #  apply the encoder layers to the input
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float) -> None:
        """
        Encoder layer for the transformer model (one encoder layer is composed of a multi-head attention layer and a position-wise fully connected feed forward layer)

        Args:
            d_model (int): dimension of the model
            num_heads (int): number of heads for the multi-head attention
            d_ffn (int): dimension of the feed forward network
            dropout (float): dropout rate

        Returns:
            None
        """
        super().__init__()
        #  set the parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.dropout = dropout
        #  initialize the multi-head attention layer and the position-wise fully connected feed forward layer
        self.multihead_attention_layer = MultiHeadAttentionSubLayer(
            d_model=self.d_model, num_heads=self.num_heads, dropout=self.dropout)
        self.position_wise_fully_connected_feed_forward_layer = PositionWiseFullyConnectedFeedForwardSubLayer(
            d_model=self.d_model, dropout=self.dropout, d_ffn=self.d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model

        Args:
            x (torch.Tensor): input of the model

        Returns:
            x (torch.Tensor): output of the model
        """

        #  apply the multi-head attention layer and the position-wise fully connected feed forward layer to the input
        x = self.multihead_attention_layer(x, x, x, None)
        x = self.position_wise_fully_connected_feed_forward_layer(x)
        return x


class MultiHeadAttentionSubLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        """
        Multi-head attention layer for the transformer model

        Args:
            d_model (int): dimension of the model
            num_heads (int): number of heads for the multi-head attention
            dropout (float): dropout rate

        Returns:
            None
        """
        super().__init__()
        #  set the parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        #  initialize the multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.num_heads, dropout=self.dropout)
        # initialize the layer normalization layer
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward pass of the model

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model]
            key (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model]
            value (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model]
            attn_mask (Any): Attention mask that indicates which entries should not be used

        Returns:
            x (torch.Tensor): output of the model
        """

        #  apply the multi-head attention layer to the query, key and value tensors
        x, _ = self.multihead_attention(query, key, value, attn_mask)
        x = self.layer_norm(query + x)
        return x


class PositionWiseFullyConnectedFeedForwardSubLayer(nn.Module):

    def __init__(self, d_model: int, dropout: float, d_ffn: int) -> None:
        """
        Position-wise fully connected feed forward layer for the transformer model

        Args:
            d_model (int): dimension of the model
            dropout (float): dropout rate
            d_ffn (int): dimension of the feed forward network

        Returns:
            None
        """

        super().__init__()
        #  set the parameters
        self.d_model = d_model
        self.dropout = dropout
        self.d_ffn = d_ffn
        # initialize the feed forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_ffn),
            nn.GELU(),
            nn.Linear(self.d_ffn, self.d_model)
        )
        #  initialize the layer normalization layer and the dropout layer
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model

        Args:
            x (torch.Tensor): input of the model

        Returns:
            x (torch.Tensor): output of the model
        """

        #  apply the feed forward layer to the input
        x = self.layer_norm(x + self.feed_forward(x))
        x = self.dropout_layer(x)
        return x


class Decoder(nn.Module):

    def __init__(self, vocab_size, max_seq_len: int, d_model: int, num_heads: int, num_layers: int, d_ffn: int, dropout: float) -> None:
        """
        Decoder model for the transformer model

        Args:
            vocab_size (int): size of the vocabulary
            max_seq_len (int): maximum length of the sequence
            d_model (int): dimension of the model
            num_heads (int): number of heads for the multi-head attention
            num_layers (int): number of layers for the encoder and the decoder
            d_ffn (int): dimension of the feed forward network
            dropout (float): dropout rate

        Returns:
            None
        """

        super().__init__()
        #  set the parameters
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ffn = d_ffn
        self.dropout = dropout
        #  initialize the positional encoding and the decoder layers
        self.positional_encoding = PositionalEncoding(
            vocab_size, d_model=self.d_model, max_seq_len=self.max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(
            self.d_model, self.num_heads, self.d_ffn, self.dropout) for _ in range(num_layers)])

    def forward(self, encoder_output: torch.Tensor, label: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model

        Args:
            encoder_output (torch.Tensor): output of the encoder
            label (torch.Tensor): target of the model
            attn_mask (torch.Tensor): attention mask of the model

        Returns:
            x (torch.Tensor): output of the model
        """

        #  add the positional encoding to the target
        label = self.positional_encoding(label)
        #  apply the decoder layers to the encoder output and the target
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(encoder_output, label, attn_mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float) -> None:
        """
        Decoder layer for the transformer model (one decoder layer is composed of a multi-head attention layer, a multi-head attention layer with a mask and a position-wise fully connected feed forward layer)

        Args:
            d_model (int): dimension of the model
            num_heads (int): number of heads for the multi-head attention
            d_ffn (int): dimension of the feed forward network
            dropout (float): dropout rate

        Returns:
            None
        """
        super().__init__()
        #  set the parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_ffn = d_ffn
        # initialize the masked multi-head attention layer
        self.mask_multihead_attention_layer = MultiHeadAttentionSubLayer(
            d_model=self.d_model, num_heads=self.num_heads, dropout=self.dropout)
        # initialize the multi-head attention layer
        self.multihead_attention_layer = MultiHeadAttentionSubLayer(
            d_model=self.d_model, num_heads=self.num_heads, dropout=self.dropout)
        #  initialize the position-wise fully connected feed forward layer
        self.position_wise_fully_connected_feed_forward_layer = PositionWiseFullyConnectedFeedForwardSubLayer(
            d_model=self.d_model, d_ffn=d_ffn, dropout=self.dropout)

    def forward(self, encoder_output: torch.Tensor, decoder_input: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model

        Args:
            encoder_output (torch.Tensor): output of the encoder
            decoder_input (torch.Tensor): input of the decoder
            attn_mask (torch.Tensor): attention mask of the model

        Returns:
            x (torch.Tensor): output of the model
        """
        #  apply the masked multi-head attention layer, the multi-head attention layer and the position-wise fully connected feed forward layer to the encoder output and the decoder input
        x = self.mask_multihead_attention_layer(
            decoder_input, decoder_input, decoder_input, attn_mask)
        x = self.multihead_attention_layer(
            encoder_output, encoder_output, x, None)
        x = self.position_wise_fully_connected_feed_forward_layer(x)
        return x


# to make a quick test of the model and avoid restarting the kernel each time we change the model
if __name__ == "__main__":

    # Initialize Oscar object
    oscar = Oscar(language="fr", split="train", max_length=200)

    # Get the dictionary of Oscar
    dictionary = list(oscar.get_vocab().keys())

    # Initialize the model
    model = Transformer(dictionary, max_seq_len=200)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Set device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    # Take a subset of the dataset (every 10th element) to make quick tests
    indices = list(range(0, len(oscar), 10))

    # Create the subset for smaller dataset
    subset = torch.utils.data.Subset(oscar, indices)

    # Define DataLoader for Oscar dataset
    batch_size = 128  # Change this to whatever fits in our GPU
    dataloader = DataLoader(subset, batch_size=batch_size,
                            shuffle=True, num_workers=6)

  
    num_epochs = 5  # Change this too if we want to train for more epochs
    best_loss = float('inf') # Set initial loss to infinity
    patience, trials = 10, 0 # Early stopping

    model.train()  # Ensure the model is in training mode
    losses = []
    total_losses = []

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()  # Reset gradients at the beginning of each epoch
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Move inputs and targets to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Create attention mask
            attn_mask = torch.full([model.max_seq_len, batch_size], -np.inf)
            attn_mask = torch.triu(attn_mask, diagonal=1).to(device)

            # Forward pass
            outputs = model(inputs, targets, attn_mask)

            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()  # Backward pass

            # Update weights and reset gradients
            optimizer.step()
            optimizer.zero_grad()

            # Update total loss
            total_loss += loss.item()
            losses.append(loss.item())
            
        total_losses.append(total_loss)
        # Update learning rate
        scheduler.step()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

        # Early stopping
        if avg_loss < best_loss:
            # if the loss is better than the best loss, save the model
            trials = 0
            best_loss = avg_loss
            torch.save(model.state_dict(), "camembert.pth")
        else:
            # if the loss is not better than the best loss, increment the number of trials
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
