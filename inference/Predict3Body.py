import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn

from simulations.BouncyBall import output_path

root = Path(__file__).resolve().parents[1]

starting_row = 50
n_frames = 100

# Neural network
model = nn.Sequential(
    nn.Linear(12, 64),
    nn.ReLU(),
    nn.Linear(64, 12),
)

weights = torch.load(root / "models" / "3body_model.pt", map_location="cpu", weights_only=True)

model.load_state_dict(weights)
model.eval()

# Read the starting point
row = np.loadtxt(
    root / "data" / "gravity_simulation_data_3body.csv",
    delimiter=",",
    skiprows=starting_row + 1,
    max_rows=1,
    dtype=np.float32
)
state = torch.from_numpy(row[:12].reshape(1,12))

frames = [state.squeeze(0).tolist()]

with torch.no_grad():
    for step in range(n_frames):
        state = model(state)

        if not torch.isfinite(state).all():
            raise ValueError(f"Non-finite prediction at step {step + 1}")

        frames.append(state.squeeze(0).tolist())

output_path = root / "predictions" / "3body_predictions.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

header = [
    "x1", "y1", "vx1", "vy1",
    "x2", "y2", "vx2", "vy2",
    "x3", "y3", "vx3", "vy3",
]

with output_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(frames)

print(f"Saved {len(frames)} frames to {output_path}")