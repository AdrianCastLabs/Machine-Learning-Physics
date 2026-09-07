import csv
import torch
import torch.nn as nn
import torch.optim as optim

epochs = 50
frames = 200

batch_frames = 100
state_size = 8

X1_START = 2
Y1_START = 2
VX1_START = -2
VY1_START = 0

X2_START = -4
Y2_START = -2
VX2_START = 1
VY2_START = 0

device = torch.device("cuda")
print(f"using device: {device}")

# load data
rows = []

with open('../data/gravity-simulation-data.csv') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        rows.append([float(x) for x in row])

data = torch.tensor(rows, dtype=torch.float32)

states = data[:, :8]

sequence_inputs = []
sequence_targets = []

frames_per_episode = 250

for episode_start in range(0, len(states), frames_per_episode):
    episode_states = states[episode_start:episode_start + frames_per_episode]

    for start in range(len(episode_states) - batch_frames):
        input_state = episode_states[start]
        future_states = episode_states[start + 1:start + 1 + batch_frames]

        sequence_inputs.append(input_state)
        sequence_targets.append(future_states - input_state)

inputs_raw = torch.stack(sequence_inputs)
targets_raw = torch.stack(sequence_targets)

inputs_mean = inputs_raw.mean(dim=0).to(device)
inputs_std = inputs_raw.std(dim=0).to(device)

targets_mean = targets_raw.mean(dim=(0, 1)).to(device)
targets_std = targets_raw.std(dim=(0, 1)).to(device)

inputs = ((inputs_raw - inputs_raw.mean(dim=0)) / inputs_raw.std(dim=0)).to(device)
targets = ((targets_raw - targets_raw.mean(dim=(0, 1))) / targets_raw.std(dim=(0, 1))).to(device)

model = nn.Sequential(
    nn.Linear(8, 128),
    nn.SiLU(),
    nn.Linear(128, 128),
    nn.SiLU(),
    nn.Linear(128, batch_frames * state_size)
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# train

for epoch in range(epochs):
    optimizer.zero_grad()

    prediction = model(inputs)
    prediction = prediction.view(-1, batch_frames, state_size)

    loss = loss_fn(prediction, targets)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss:.6f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "batch_frames": batch_frames,
    "state_size": state_size,
    "inputs_mean": inputs_mean.cpu(),
    "inputs_std": inputs_std.cpu(),
    "targets_mean": targets_mean.cpu(),
    "targets_std": targets_std.cpu(),
}, "model_checkpoint.pt")

print("saved model_checkpoint.pt")

state = torch.tensor([[X1_START, Y1_START, VX1_START, VY1_START, X2_START, Y2_START, VX2_START, VY2_START]],
                      dtype=torch.float32, device=device)
generated = []
previous_state = state[0]

with torch.no_grad():
    for _ in range(frames):
        state_normalized = (state - inputs_mean) / inputs_std

        delta_normalized = model(state_normalized)
        delta_normalized = delta_normalized.view(batch_frames, state_size)

        delta = delta_normalized * targets_std + targets_mean
        predicted_states = state + delta

        for next_state in predicted_states:
            row = previous_state.tolist() + next_state.tolist()
            generated.append(row)
            previous_state = next_state

with open('../data/gravity-simulation-predictions.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2", "x1_next", "y1_next", "vx1_next", "vy1_next", "x2_next", "y2_next", "vx2_next", "vy2_next"])
    for row in generated:
        writer.writerow([f"{val:.4f}" for val in row])