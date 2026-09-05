import csv
import random
import numpy as np

rng = np.random.default_rng()

g = 100
dt = 0.02
num_episodes = 50
frames_per_episode = 250
spawnArea = 5
random_velocity = 2

data = [] # rows: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x1_next, y1_next, vx1_next, vy1_next, x2_next, y2_next, vx2_next, vy2_next]

for episode in range(num_episodes):
    print(episode)

    positions = []
    velocities = []

    positions.append(rng.uniform(low = -spawnArea, high = spawnArea, size = 2))
    positions.append(rng.uniform(low = -spawnArea, high = spawnArea, size = 2))

    velocities.append(rng.uniform(low = -random_velocity, high = random_velocity, size = 2))
    velocities.append(rng.uniform(low = -random_velocity, high = random_velocity, size = 2))

    for frame in range(frames_per_episode):
        x1_prev = positions[0][0]
        y1_prev = positions[0][1]
        vx1_prev = velocities[0][0]
        vy1_prev = velocities[0][1]
        x2_prev = positions[1][0]
        y2_prev = positions[1][1]
        vx2_prev = velocities[1][0]
        vy2_prev = velocities[1][1]

        for i in range (len(positions)):
            for j in range(len(positions)):
                if i == j: continue

                direction = positions[j] - positions[i]
                distance = np.linalg.norm(direction)
                distance += 1
                direction_normalized = direction / distance
                force_magnitude = g / distance**2

                velocities[i] += force_magnitude * direction_normalized * dt
                positions[i] += velocities[i] * dt

        data.append([
            x1_prev, y1_prev, vx1_prev, vy1_prev,
            x2_prev, y2_prev, vx2_prev, vy2_prev,
            positions[0][0], positions[0][1],
            velocities[0][0], velocities[0][1],
            positions[1][0], positions[1][1],
            velocities[1][0], velocities[1][1]])

with open('../data/gravity-simulation-data.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2", "x1_next", "y1_next", "vx1_next", "vy1_next", "x2_next", "y2_next", "vx2_next", "vy2_next"])
    for row in data:
        writer.writerow([f"{val:.4f}" for val in row])