import csv
import sys
import pygame

# config
WIDTH, HEIGHT = 400, 600
FLOOR_Y = 0.5
WORLD_TOP = 11.0
BALL_RADIUS = 15
FPS = 60
FRAMES_PER_EPISODE = 150

def world_to_screen_y(y):
    t = y / WORLD_TOP
    return int(HEIGHT - t * HEIGHT)

def load_y_values(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # pick whichever column represents position
        for candidate in ["y", "predicted_y", "y_next"]:
            if candidate in headers:
                col = candidate
                break
        else:
            raise ValueError(f"No recognizable y column found in {headers}")

        return [float(row[col]) for row in reader]

def main():
    path = "../data/bouncy-ball-predictions.csv"
    y_values = load_y_values(path)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Bouncy Ball Viewer - {path}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if frame >= len(y_values):
            frame = 0  # loop back to start

        y = y_values[frame]
        screen_y = world_to_screen_y(y)

        screen.fill((30, 30, 30))
        pygame.draw.line(screen, (80, 80, 80), (0, HEIGHT - 1), (WIDTH, HEIGHT - 1), 2)
        pygame.draw.circle(screen, (255, 100, 100), (WIDTH // 2, screen_y), BALL_RADIUS)

        episode = frame // FRAMES_PER_EPISODE
        label = font.render(f"frame {frame}  episode {episode}  y={y:.2f}", True, (255, 255, 255))
        screen.blit(label, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()

if __name__ == "__main__":
    main()