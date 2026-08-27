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

data = torch.tensor(rows)
X = data[:, :2]
Y = data[:, 2:]

# model
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# train
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss}:.6f")

state = torch.tensor([[START_Y, START_V]])
generated = []

with torch.no_grad():
    for _ in range(N_FRAMES):
        y_prev, v_prev = state[0].tolist()
        next_state = model(state)
        y_next, v_next = next_state[0].tolist()

        generated.append([y_prev, v_prev, y_next, v_next])
        state = next_state

with open('../data/bouncy-ball-predictions.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["y", "v", "y_next", "v_next"])
    for row in generated:
        writer.writerow([f"{val:.4f}" for val in row])