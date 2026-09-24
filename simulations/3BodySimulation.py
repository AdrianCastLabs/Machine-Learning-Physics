import csv
import numpy as np
from pathlib import Path

rng = np.random.default_rng()

g = 100
dt = 0.02
num_episodes = 2000
frames_per_episode = 250
spawnArea = 5
random_velocity = 2
num_bodies = 3

output_path = Path('../data/gravity_simulation_data_3body.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)

# columns: for each body i: xi, yi, vxi, vyi  (prev), then same set "_next"
header = []
for i in range(1, num_bodies + 1):
    header += [f"x{i}", f"y{i}", f"vx{i}", f"vy{i}"]
for i in range(1, num_bodies + 1):
    header += [f"x{i}_next", f"y{i}_next", f"vx{i}_next", f"vy{i}_next"]

data = []

for episode in range(num_episodes):
    print(episode)

    positions = [rng.uniform(low=-spawnArea, high=spawnArea, size=2) for _ in range(num_bodies)]
    velocities = [rng.uniform(low=-random_velocity, high=random_velocity, size=2) for _ in range(num_bodies)]

    for frame in range(frames_per_episode):
        prev_positions = [p.copy() for p in positions]
        prev_velocities = [v.copy() for v in velocities]

        for i in range(num_bodies):
            for j in range(num_bodies):
                if i == j:
                    continue

                direction = positions[j] - positions[i]
                distance = np.linalg.norm(direction)
                distance += 1
                direction_normalized = direction / distance
                force_magnitude = g / distance**2

                velocities[i] += force_magnitude * direction_normalized * dt
                positions[i] += velocities[i] * dt

        row = []
        for i in range(num_bodies):
            row += [prev_positions[i][0], prev_positions[i][1],
                    prev_velocities[i][0], prev_velocities[i][1]]
        for i in range(num_bodies):
            row += [positions[i][0], positions[i][1],
                    velocities[i][0], velocities[i][1]]
        data.append(row)

with open(output_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in data:
        writer.writerow([f"{val:.4f}" for val in row])