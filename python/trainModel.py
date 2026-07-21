import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

# load csv
df = pd.read_csv("simulation_data.csv")

print(f"loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

data = df.values.astype(np.float32)

# normalize
#mean = data.mean(axis=0)
#std = data.std(axis=0)

#std[std == 0] = 1

#data = (data - mean) / std

# build input/output pairs
inputs = data[:-1]
outputs = data[1:]

print(f"training pairs: {len(inputs)}")

# pytorch dataset
class SimDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.X = torch.tensor(inputs, dtype=torch.float32)
        self.y = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = SimDataset(inputs, outputs)

dataloader = DataLoader(dataset, batch_size=32768, shuffle=True, pin_memory=True)

# define the network

N_BODIES = 3
N_INPUT = N_BODIES * 4
N_OUTPUT = N_BODIES * 4

class GravityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, N_OUTPUT)
        )

    def forward(self, x):
        return self.net(x)

model = GravityNet().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
EPOCHS = 100

lossHistory = []

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch_inputs, batch_outputs in dataloader:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_outputs = batch_outputs.to(device, non_blocking=True)

        predictions = model(batch_inputs)
        loss = loss_fn(predictions, batch_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    lossHistory.append(total_loss)
    avg_loss = total_loss / len(dataloader)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6F}")

# save the model
torch.save(model.state_dict(), "gravity_model.pth")
print("\nmodel saved to gravity_model.pth")

plt.plot(lossHistory)
plt.show()













