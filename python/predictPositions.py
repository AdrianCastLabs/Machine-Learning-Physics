import torch
import torch.nn as nn
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = GravityNet().to(device)
model.load_state_dict(torch.load("gravity_model.pth", map_location=device))
model.eval()

df = pd.read_csv("simulation_data.csv")

current_state = torch.tensor(
    df.iloc[499].values.astype(np.float32),
    device=device
).unsqueeze(0)

predictions = []

with torch.no_grad():
    for i in range(500):

        current_state = model(current_state)

        predictions.append(
            current_state.squeeze(0).cpu().numpy()
        )

pred_df = pd.DataFrame(predictions, columns=df.columns)
pred_df.to_csv("predicted_positions.csv", index=False)

print("Saved predicted positions")