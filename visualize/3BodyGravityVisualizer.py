import csv
import pygame

# config
WIDTH, HEIGHT = 900, 900
FPS = 60
FRAMES_PER_EPISODE = 500

WORLD_MIN = -30.0
WORLD_MAX = 30.0
BODY_RADIUS = 8

BODY_COLORS = [(100, 180, 255), (255, 140, 100), (140, 255, 140)]
TRAIL_COLORS = [(60, 110, 160), (160, 80, 60), (80, 160, 80)]
BACKGROUND_COLOR = (25, 25, 30)
AXIS_COLOR = (70, 70, 80)
TEXT_COLOR = (255, 255, 255)

TRAIL_LENGTH = 0
NUM_BODIES = 3

path = "../data/gravity_simulation_data_3body.csv"


def world_to_screen(x, y):
    t_x = (x - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)
    t_y = (y - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)

    screen_x = int(t_x * WIDTH)
    screen_y = int(HEIGHT - t_y * HEIGHT)

    return screen_x, screen_y


def load_positions(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        required_columns = []
        for i in range(1, NUM_BODIES + 1):
            required_columns += [f"x{i}", f"y{i}"]
        for column in required_columns:
            if column not in headers:
                raise ValueError(f"Missing required column '{column}' in {headers}")

        positions = []
        for row in reader:
            frame_positions = []
            for i in range(1, NUM_BODIES + 1):
                frame_positions.append(float(row[f"x{i}"]))
                frame_positions.append(float(row[f"y{i}"]))
            positions.append(tuple(frame_positions))

        return positions


def draw_axes(screen):
    origin_x, origin_y = world_to_screen(0, 0)

    pygame.draw.line(screen, AXIS_COLOR, (0, origin_y), (WIDTH, origin_y), 1)
    pygame.draw.line(screen, AXIS_COLOR, (origin_x, 0), (origin_x, HEIGHT), 1)


def draw_trail(screen, trail, color):
    if len(trail) < 2:
        return

    points = [world_to_screen(x, y) for x, y in trail]
    pygame.draw.lines(screen, color, False, points, 2)


def main():
    positions = load_positions(path)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Gravity Simulation Viewer - {path}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    frame = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if frame >= len(positions):
            frame = 0

        current = positions[frame]
        bodies = [(current[2 * i], current[2 * i + 1]) for i in range(NUM_BODIES)]

        start = max(0, frame - TRAIL_LENGTH)
        recent_positions = positions[start:frame + 1]

        trails = []
        for i in range(NUM_BODIES):
            trails.append([(row[2 * i], row[2 * i + 1]) for row in recent_positions])

        screen.fill(BACKGROUND_COLOR)
        draw_axes(screen)

        for i in range(NUM_BODIES):
            draw_trail(screen, trails[i], TRAIL_COLORS[i % len(TRAIL_COLORS)])

        for i in range(NUM_BODIES):
            sx, sy = world_to_screen(bodies[i][0], bodies[i][1])
            pygame.draw.circle(screen, BODY_COLORS[i % len(BODY_COLORS)], (sx, sy), BODY_RADIUS)

        episode = frame // FRAMES_PER_EPISODE
        body_str = "  ".join(f"body{i+1}=({bodies[i][0]:.2f}, {bodies[i][1]:.2f})" for i in range(NUM_BODIES))
        label = font.render(f"frame {frame}  episode {episode}  {body_str}", True, TEXT_COLOR)
        screen.blit(label, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()


if __name__ == "__main__":
    main()