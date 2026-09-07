import csv
import torch
import torch.nn as nn
import torch.optim as optim

epochs = 2000
batch_frames = 100
state_size = 8

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

inputs_mean = inputs_raw.mean(dim=0)
inputs_std = inputs_raw.std(dim=0)
targets_mean = targets_raw.mean(dim=(0, 1))
targets_std = targets_raw.std(dim=(0, 1))

inputs = ((inputs_raw - inputs_mean) / inputs_std).to(device)
targets = ((targets_raw - targets_mean) / targets_std).to(device)
inputs_mean, inputs_std = inputs_mean.to(device), inputs_std.to(device)
targets_mean, targets_std = targets_mean.to(device), targets_std.to(device)

model = nn.Sequential(
    nn.Linear(8, 128),
    nn.SiLU(),
    nn.Linear(128, 128),
    nn.SiLU(),
    nn.Linear(128, batch_frames * state_size)
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

scaler_dtype = torch.bfloat16

# train

for epoch in range(epochs):
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=scaler_dtype):
        prediction = model(inputs)
        prediction = prediction.view(-1, batch_frames, state_size)
        loss = loss_fn(prediction, targets)

    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print(f"epoch {epoch}, loss {loss:.6f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "batch_frames": batch_frames,
    "state_size": state_size,
    "inputs_mean": inputs_mean.cpu(),
    "inputs_std": inputs_std.cpu(),
    "targets_mean": targets_mean.cpu(),
    "targets_std": targets_std.cpu(),
}, "../models/gravity_model_checkpoint.pt")

print("saved gravity_model_checkpoint.pt")

