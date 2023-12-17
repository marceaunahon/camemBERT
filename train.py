import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from roberta import Roberta
from oscar import Oscar
from model import Transformer
import numpy as np
from matplotlib import pyplot as plt
import random as rd
from torch.optim import Adam
from torch.nn import GELU
from transformers import get_linear_schedule_with_warmup

# Initialize Oscar object
oscar = Oscar(language="fr", split="train", max_length=200)

# Define Roberta model
dictionary = list(oscar.get_vocab().keys())
# model_name = "roberta-base"  
# model = Roberta(model_name) # si on utilise Roberta : faire un controle h et remplacer tous les models.parameters() par model.model.parameters()
model = Transformer(dictionary,max_seq_len=200) # si on utilise model : faire un controle h et remplacer tous les models.models.parameters() par model.parameters()

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
#

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

indices = list(range(0, len(oscar), 10))  # This will take every second element

# Create the subset for smaller dataset
subset = torch.utils.data.Subset(oscar, indices)

# Define DataLoader for Oscar dataset
batch_size = 128 # Change this to whatever fits in our GPU
dataloader = DataLoader(oscar, batch_size=batch_size, shuffle=True, num_workers=6)
#dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=6)


# Define the optimizer
optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

# Define the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=1000000)

# Training loop
num_epochs = 5 # Change this too if we want to train for more epochs
best_loss = float('inf')
patience, trials = 10, 0

losses = []
total_losses = []

model.train()  # Ensure the model is in training mode
for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()  # Reset gradients at the beginning of each epoch
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs, targets = inputs.to(device), targets.to(device)

        attn_mask = torch.full([model.max_seq_len, len(inputs)], -np.inf)
        attn_mask = torch.triu(attn_mask, diagonal=1).to(device)

        # Forward pass
        outputs = model(inputs, targets, attn_mask)

        # Calculate loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()  # Backward pass

        # Perform optimization step only after accumulating gradients for accumulation_steps batches
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        losses.append(loss.item())

    total_losses.append(total_loss)
    scheduler.step()

    # Save the loss plot to the 'loss' directory and display it
    plt.figure()
    plt.plot(losses)
    plt.savefig(f'loss/loss_plot_epoch_{epoch+1}.png')
    plt.show()
    
    # Print average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    # Early stopping
    if avg_loss < best_loss:
        trials = 0
        best_loss = avg_loss
        torch.save(model.state_dict(), "robertaparam.pth")
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break


