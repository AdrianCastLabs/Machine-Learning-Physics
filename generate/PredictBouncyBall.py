import csv
import random
import torch
import torch.nn as nn

frames = 150

START_Y = random.randrange(5, 10)
START_V = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('../models/bouncy_ball_model_checkpoint.pt', map_location=device)

inputs_mean = checkpoint["inputs_mean"]
inputs_std = checkpoint["inputs_std"]
targets_mean = checkpoint["targets_mean"]
targets_std = checkpoint["targets_std"]

# rebuild model
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# load model
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

state = torch.tensor([[START_Y, START_V]], dtype=torch.float32)
generated = []

floor_y = 0.5

with torch.no_grad():
    for _ in range(frames):
        y_prev, v_prev = state[0].tolist()

        state_normalized = (state - inputs_mean) / inputs_std
        delta_normalized = model(state_normalized)
        delta = delta_normalized * targets_std + targets_mean

        next_state = state + delta
        y_next, v_next = next_state[0].tolist()

        if y_next < floor_y:
            y_next = floor_y

        generated.append([y_prev, v_prev, y_next, v_next])
        state = torch.tensor([[y_next, v_next]], dtype=torch.float32)

        print(f"frame: {_}")

with open('../predictions/bouncy_ball_predictions.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["y", "v", "y_next", "v_next"])
    for row in generated:
        writer.writerow([f"{val:.4f}" for val in row])

print("saved predictions")