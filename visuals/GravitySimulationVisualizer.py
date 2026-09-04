import csv
import pygame

# config
WIDTH, HEIGHT = 700, 700
FPS = 60
FRAMES_PER_EPISODE = 500

WORLD_MIN = -6.0
WORLD_MAX = 6.0
BODY_RADIUS = 8

BODY_1_COLOR = (100, 180, 255)
BODY_2_COLOR = (255, 140, 100)
TRAIL_COLOR_1 = (60, 110, 160)
TRAIL_COLOR_2 = (160, 80, 60)
BACKGROUND_COLOR = (25, 25, 30)
AXIS_COLOR = (70, 70, 80)
TEXT_COLOR = (255, 255, 255)

TRAIL_LENGTH = 120


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

        required_columns = ["x1", "y1", "x2", "y2"]
        for column in required_columns:
            if column not in headers:
                raise ValueError(f"Missing required column '{column}' in {headers}")

        positions = []
        for row in reader:
            positions.append(
                (
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                )
            )

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
    path = "../data/gravity-simulation-data.csv"
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

        x1, y1, x2, y2 = positions[frame]

        start = max(0, frame - TRAIL_LENGTH)
        recent_positions = positions[start:frame + 1]

        trail_1 = [(row[0], row[1]) for row in recent_positions]
        trail_2 = [(row[2], row[3]) for row in recent_positions]

        screen.fill(BACKGROUND_COLOR)
        draw_axes(screen)

        draw_trail(screen, trail_1, TRAIL_COLOR_1)
        draw_trail(screen, trail_2, TRAIL_COLOR_2)

        screen_x1, screen_y1 = world_to_screen(x1, y1)
        screen_x2, screen_y2 = world_to_screen(x2, y2)

        pygame.draw.circle(screen, BODY_1_COLOR, (screen_x1, screen_y1), BODY_RADIUS)
        pygame.draw.circle(screen, BODY_2_COLOR, (screen_x2, screen_y2), BODY_RADIUS)

        episode = frame // FRAMES_PER_EPISODE
        label = font.render(
            f"frame {frame}  episode {episode}  "
            f"body1=({x1:.2f}, {y1:.2f})  body2=({x2:.2f}, {y2:.2f})",
            True,
            TEXT_COLOR,
        )
        screen.blit(label, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()


if __name__ == "__main__":
    main()