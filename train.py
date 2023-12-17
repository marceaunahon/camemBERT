import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from oscar import Oscar
from model import Transformer
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam

# Initialize Oscar object
oscar = Oscar(language="fr", split="train", max_length=200)

# Get the dictionary of Oscar
dictionary = list(oscar.get_vocab().keys())

# Initialize the model
model = Transformer(dictionary, max_seq_len=200)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

# Take a subset of the dataset (every 10th element) to make quick tests
indices = list(range(0, len(oscar), 1000))

# Create the subset for smaller dataset
subset = torch.utils.data.Subset(oscar, indices)

# Define DataLoader for Oscar dataset
batch_size = 8  # Change this to whatever fits in our GPU
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

    # Save the loss plot to the 'loss' directory
    plt.figure()
    plt.plot(losses)
    plt.show()

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