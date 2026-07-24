import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# load csv
df = pd.read_csv("simulation_data.csv")

print(f"loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

data = df.values.astype(np.float32)

episode_length = 100

n = len(data)
frame_in_episode = np.arange(n) % episode_length
valid_mask = frame_in_episode[:-1] != (episode_length - 1)

# build input/output pairs
inputs = data[:-1][valid_mask]
outputs = data[1:][valid_mask] - data[:-1][valid_mask]

print(f"training pairs: {len(inputs)}")

# normalize
input_mean = inputs.mean(axis=0)
input_std = inputs.std(axis=0) + 1e-8
delta_mean = outputs.mean(axis=0)
delta_std = outputs.std(axis=0) + 1e-8

inputs_norm = (inputs - input_mean) / input_std
outputs_norm = (outputs - delta_mean) / delta_std

np.save("input_mean.npy", input_mean)
np.save("input_std.npy", input_std)
np.save("delta_mean.npy", delta_mean)
np.save("delta_std.npy", delta_std)

# pytorch dataset
class SimDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.X = torch.tensor(inputs, dtype=torch.float32)
        self.y = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = SimDataset(inputs_norm, outputs_norm)

dataloader = DataLoader(dataset, batch_size=32768, shuffle=True)

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

model = GravityNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
EPOCHS = 300

lossHistory = []

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch_inputs, batch_outputs in dataloader:
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
