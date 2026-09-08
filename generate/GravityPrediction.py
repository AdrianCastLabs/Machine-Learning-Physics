import csv
import random
import torch
import torch.nn as nn

frames = 400

X1_START = random.randrange(-4, 4)
Y1_START = random.randrange(-4, 4)
VX1_START = random.randrange(-4, 4)
VY1_START = random.randrange(-4, 4)

X2_START = random.randrange(-4, 4)
Y2_START = random.randrange(-4, 4)
VX2_START = random.randrange(-4, 4)
VY2_START = random.randrange(-4, 4)

device = torch.device("cuda")
print(f"using device: {device}")

# load checkpoint
checkpoint = torch.load('../models/gravity_model_checkpoint.pt', map_location=device)

batch_frames = checkpoint["batch_frames"]
state_size = checkpoint["state_size"]

inputs_mean = checkpoint["inputs_mean"].to(device)
inputs_std = checkpoint["inputs_std"].to(device)
targets_mean = checkpoint["targets_mean"].to(device)
targets_std = checkpoint["targets_std"].to(device)

# rebuild model
model = nn.Sequential(
    nn.Linear(8, 128),
    nn.SiLU(),
    nn.Linear(128, 128),
    nn.SiLU(),
    nn.Linear(128, batch_frames * state_size)
).to(device)

# load model
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# generate
state = torch.tensor([[X1_START, Y1_START, VX1_START, VY1_START, X2_START, Y2_START, VX2_START, VY2_START]], dtype=torch.float32, device=device)
generated = []
previous_state = state[0]

with torch.no_grad():
    while len(generated) < frames:
        state_normalized = (state - inputs_mean) / inputs_std

        delta_normalized = model(state_normalized)
        delta_normalized = delta_normalized.view(batch_frames, state_size)

        delta = delta_normalized * targets_std + targets_mean
        predicted_states = state + delta

        for next_state in predicted_states:
            if len(generated) >= frames:
                break

            row = previous_state.tolist() + next_state.tolist()
            generated.append(row)
            previous_state = next_state

        state = predicted_states[-1].unsqueeze(0)

with open('../predictions/gravity_simulation_predictions.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2", "x1_next", "y1_next", "vx1_next", "vy1_next", "x2_next",
         "y2_next", "vx2_next", "vy2_next"])
    for row in generated:
        writer.writerow([f"{val:.4f}" for val in row])

print("saved predictions")
