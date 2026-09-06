import csv
import torch
import torch.nn as nn
import torch.optim as optim

epochs = 5000
frames = 150

X1_START = 1
Y1_START = 2
VX1_START = 1
VY1_START = 2

X2_START = -1
Y2_START = -2
VX2_START = -1
VY2_START = 0

# load data
rows = []

with open('../data/gravity-simulation-data.csv') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        rows.append([float(x) for x in row])

data = torch.tensor(rows, dtype=torch.float32)

inputs_raw = data[:, :8]
targets_raw = data[:, 8:] - data[:, :8]

inputs_mean = inputs_raw.mean(dim=0)
inputs_std = inputs_raw.std(dim=0)
targets_mean = targets_raw.mean(dim=0)
targets_std = targets_raw.std(dim=0)

inputs = (inputs_raw - inputs_mean) / inputs_std
targets = (targets_raw - targets_mean) / targets_std

model = nn.Sequential(
    nn.Linear(8, 128),
    nn.SiLU(),
    nn.Linear(128, 128),
    nn.SiLU(),
    nn.Linear(128, 8)
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# train

for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(inputs)
    loss = loss_fn(prediction, targets)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss}:.6f")

state = torch.tensor([[X1_START, Y1_START, VX1_START, VY1_START, X2_START, Y2_START, VX2_START, VY1_START]], dtype=torch.float32)
generated = []

with torch.no_grad():
    for _ in range(frames):
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state[0].tolist()

        state_normalized = (state - inputs_mean) / inputs_std
        delta_normalized = model(state_normalized)
        delta = delta_normalized * targets_std + targets_mean

        next_state = state + delta
        x1_next, y1_next, vx1_next, vy1_next, x2_next, y2_next, vx2_next, vy2_next = next_state[0].tolist()

        generated.append([x1, y1, vx1, vy1, x2, y2, vx2, vy2, x1_next, y1_next, vx1_next, vy1_next, x2_next, y2_next, vx2_next, vy2_next])
        state = torch.tensor([[x1_next, y1_next, vx1_next, vy1_next, x2_next, y2_next, vx2_next, vy2_next]], dtype=torch.float32)

with open('../data/gravity-simulation-predictions.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2", "x1_next", "y1_next", "vx1_next", "vy1_next", "x2_next", "y2_next", "vx2_next", "vy2_next"])
    for row in generated:
        writer.writerow([f"{val:.4f}" for val in row])
