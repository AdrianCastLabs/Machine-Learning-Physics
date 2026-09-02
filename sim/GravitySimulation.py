import random
import numpy as np

rng = np.random.default_rng()

g = 1
dt = 0.02
num_episodes = 100
frames_per_episode = 500
spawnArea = 5

data = [] # rows: [x1, y1, v1, v2, x2, y2, v1, v2]

for episode in range(num_episodes):
    positions = []
    velocities = []

    positions.append(rng.uniform(low = -spawnArea, high = spawnArea, size = 2))
    positions.append(rng.uniform(low = -spawnArea, high = spawnArea, size = 2))

    velocities.append(np.zeros(2))
    velocities.append(np.zeros(2))

    for frame in range(frames_per_episode):
        for i in range (len(positions)):
            for j in range(len(positions)):
                if i == j: continue

                direction = positions[j] - positions[i]
                distance = np.linalg.norm(direction)
                direction_normalized = direction / distance
                force_magnitude = g / distance**2

                velocities[i] += force_magnitude * direction_normalized * dt
                positions[i] += velocities[i] * dt

        data.append([
            positions[0][0], positions[0][1],
            velocities[0][0], velocities[0][1],
            positions[1][0], positions[1][1],
            velocities[1][0], velocities[1][1]])

print(data)