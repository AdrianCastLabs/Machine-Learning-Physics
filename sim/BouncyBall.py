import csv
import os
import random

# simulation params
g = -9.8
dt = 0.04
bounciness = 0.6
frames_per_episode = 200
num_episodes = 50

data = []  # rows: [y, v, y_next, v_next]

for ep in range(num_episodes):
    y = random.uniform(2.0, 10.0)  # random starting height
    v = 0.0

    for i in range(frames_per_episode):
        y_prev, v_prev = y, v

        v += g * dt
        y += v * dt

        if y <= 0.5:
            y = 0.5
            v = -v * bounciness
            if abs(v) < abs(g * dt):
                v = 0.0

        data.append([y_prev, v_prev, y, v])

os.makedirs('../data', exist_ok=True)

with open('../data/BouncyBallData.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["y", "v", "y_next", "v_next"])
    for row in data:
        writer.writerow([f"{val:.4f}" for val in row])