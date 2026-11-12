from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Load the CSV
root = Path(__file__).resolve().parents[1]
csv_path = root / "data" / "gravity_simulation_data_3body.csv"

data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)

# Select current states and next positions
inputs = torch.from_numpy(data[:, :12])
targets = torch.from_numpy(data[:, 12:])

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=256, shuffle = True)

# Neural network
model = nn.Sequential(
    nn.Linear(12, 64),
    nn.ReLU(),
    nn.Linear(64, 12),
)

# Measure error and update the network
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the network
# 　　.o　O　　0　　　O
# 　┌┐____
# 　_|  |_|□　 | ┌════┐
# (　=\  |══|=▏□　　□▕
# /_⊙⊙═◎◎╙⊙⊙═⊙⊙┘ . . . . . .

for epoch in range(20):
    total_loss = 0.0

    for current_states, next_states in loader:
        optimizer.zero_grad()

        predictions = model(current_states)
        loss = loss_function(predictions, next_states)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(current_states)

    average_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1}: loss = {average_loss:.6f}")

