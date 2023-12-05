import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your Oscar class and Transformer model
from oscar import Oscar
from model import Transformer

# Initialize Oscar object
oscar = Oscar(language="fr", split="train")

# Define Transformer model
dictionary = list(oscar.get_vocab().keys())
model = Transformer(dictionary)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define DataLoader for Oscar dataset
batch_size = 32  # Change this to whatever fits in our GPU
dataloader = DataLoader(range(len(oscar)), batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 5 # Change this to if we want to train for more epochs
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Get batch data from the Oscar dataset
        inputs = [oscar[i] for i in batch_idx]
        inputs = torch.tensor(inputs).to(device)

        # Forward pass
        outputs = model(inputs, output_encoding=None)  # Assuming output_encoding is None for training

        # Calculate loss
        targets = torch.randint(0, len(dictionary), (batch_size,)).to(device)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

# Save the trained model
torch.save(model.state_dict(), "camembert_model.pth")
