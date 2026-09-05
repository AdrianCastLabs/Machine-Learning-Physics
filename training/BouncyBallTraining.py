import csv
import torch
import torch.nn as nn

N_FRAMES = 200
START_Y = 8.0
START_V = 0.0

# load data
rows = []
with open('../data/bouncy-ball-training-data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        rows.append([float(x) for x in row])
    
data = torch.tensor(rows, dtype=torch.float32)
X_raw = data[:, :2]
Y_raw = data[:, 2:] - data[:, :2]

X_mean = X_raw.mean(dim=0)
X_std = X_raw.std(dim=0)
Y_mean = Y_raw.mean(dim=0)
Y_std = Y_raw.std(dim=0)

X = (X_raw - X_mean) / X_std
Y = (Y_raw - Y_mean) / Y_std

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
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss}:.6f")

state = torch.tensor([[START_Y, START_V]], dtype=torch.float32)
generated = []

floor_y = 0.5

with torch.no_grad():
    for _ in range(N_FRAMES):
        y_prev, v_prev = state[0].tolist()

        state_normalized = (state - X_mean) / X_std
        delta_normalized = model(state_normalized)
        delta = delta_normalized * Y_std + Y_mean

        next_state = state + delta
        y_next, v_next = next_state[0].tolist()

        if y_next < floor_y:
            y_next = floor_y

        generated.append([y_prev, v_prev, y_next, v_next])
        state = torch.tensor([[y_next, v_next]], dtype=torch.float32)

with open('../data/bouncy-ball-predictions.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["y", "v", "y_next", "v_next"])
    for row in generated:
        writer.writerow([f"{val:.4f}" for val in row])