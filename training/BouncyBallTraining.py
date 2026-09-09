import csv
import torch
import torch.nn as nn

epochs = 2000
N_FRAMES = 200

# load data
rows = []

with open('../data/bouncy_ball_simulation_data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        rows.append([float(x) for x in row])
    
data = torch.tensor(rows, dtype=torch.float32)
inputs_raw = data[:, :2]
targets_raw = data[:, 2:] - data[:, :2]

inputs_mean = inputs_raw.mean(dim=0)
inputs_std = inputs_raw.std(dim=0)
targets_mean = targets_raw.mean(dim=0)
targets_std = targets_raw.std(dim=0)

X = (inputs_raw - inputs_mean) / inputs_std
Y = (targets_raw - targets_mean) / targets_std

# model
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# train
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss}:.6f")

torch.save({
        "model_state_dict": model.state_dict(),
        "inputs_mean": inputs_mean.cpu(),
        "inputs_std": inputs_std.cpu(),
        "targets_mean": targets_mean.cpu(),
        "targets_std": targets_std.cpu()
    }, "../models/bouncy_ball_model_checkpoint.pt")

print("saved bouncy_ball_model_checkpoint.pt")

