import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ball properties
y = 8.0
v = 0.0
g = -9.8
dt = 0.02
bounciness = 0.8

fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(0, 10)
ax.set_aspect("equal")

ball, = ax.plot(0, y, "o", markersize=20)

def update(frame):
    global y, v

    v += g * dt
    y += v * dt

    if y <= 0.5:
        y = 0.5
        v = -v * bounciness

    ball.set_data([0], [y])
    return ball,

animation = FuncAnimation(fig, update, frames=500, interval=20, blit=True)
plt.show()