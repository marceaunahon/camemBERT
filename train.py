import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from roberta import Roberta
from oscar import Oscar
from model import Transformer
import numpy as np


# Initialize Oscar object
oscar = Oscar(language="fr", split="train")

# Define Roberta model
dictionary = list(oscar.get_vocab().keys())

model = Transformer(dictionary, num_layers=1) # si on utilise model : faire un controle h et remplacer tous les models.models.parameters() par model.parameters()

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
dataloader = DataLoader(oscar, batch_size=batch_size, shuffle=True, num_workers=4)

# Training loop
num_epochs = 1 # Change this too if we want to train for more epochs
best_loss = float('inf')
patience, trials = 10, 0
accumulation_steps = 4  # Change this to the number of batches to accumulate gradients

model.train()  # Ensure the model is in training mode
for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()  # Reset gradients at the beginning of each epoch
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs, targets)
        
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
        torch.save(model.state_dict(), "roberta_model.pth")
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break
