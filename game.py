"""A minimal top-down racing game controlled by hand tracking via the bridge.

This is the "receiving end" for the Virtual Steering Wheel project. Run this
game with the launcher.py script; it reads control inputs from the shared
bridge file written by main.py (the hand-tracking script).
Keyboard fallback also works: A/D to steer, W/S for gas/brake.
"""

import random
import sys
import time

import pygame

from controller_bridge import read_control

WINDOW_WIDTH = 480
WINDOW_HEIGHT = 720
ROAD_WIDTH = 320
ROAD_LEFT = (WINDOW_WIDTH - ROAD_WIDTH) // 2
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH

LANE_COUNT = 3
LANE_WIDTH = ROAD_WIDTH // LANE_COUNT

CAR_WIDTH = 46
CAR_HEIGHT = 82

PLAYER_MAX_SPEED = 5.0
PLAYER_ACCELERATION = 0.25
PLAYER_BRAKE_DECEL = 0.6
PLAYER_FRICTION = 0.08
STEER_SPEED = 8.0

TRAFFIC_MIN_SPEED = 4.0
TRAFFIC_MAX_SPEED = 7.0
TRAFFIC_SPAWN_MS = 900

WHITE = (245, 245, 245)
GRAY_ROAD = (55, 55, 60)
GRAY_SHOULDER = (30, 90, 40)
YELLOW = (230, 200, 40)
RED = (220, 60, 60)
BLUE = (60, 140, 230)
BLACK = (15, 15, 15)


class Car:
    """A rectangle-shaped car used for both the player and traffic."""

    def __init__(self, x: float, y: float, color: tuple[int, int, int]):
        self.x = x
        self.y = y
        self.color = color
        self.rect = pygame.Rect(0, 0, CAR_WIDTH, CAR_HEIGHT)
        self.rect.center = (int(x), int(y))

    def update_rect(self) -> None:
        self.rect.center = (int(self.x), int(self.y))

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.color, self.rect, border_radius=8)
        # Windshield accent so the car has a visible "front".
        windshield = pygame.Rect(0, 0, CAR_WIDTH - 16, 18)
        windshield.center = (self.rect.centerx, self.rect.top + 24)
        pygame.draw.rect(surface, (20, 20, 25), windshield, border_radius=4)


def draw_road(surface: pygame.Surface, scroll_offset: float) -> None:
    surface.fill(GRAY_SHOULDER)
    pygame.draw.rect(
        surface, GRAY_ROAD, (ROAD_LEFT, 0, ROAD_WIDTH, WINDOW_HEIGHT)
    )

    # Lane dividers that scroll downward to sell the sense of speed.
    dash_height = 40
    gap = 30
    period = dash_height + gap
    start_y = -(int(scroll_offset) % period)
    for lane in range(1, LANE_COUNT):
        x = ROAD_LEFT + lane * LANE_WIDTH
        y = start_y
        while y < WINDOW_HEIGHT:
            pygame.draw.rect(surface, YELLOW, (x - 3, y, 6, dash_height))
            y += period

    pygame.draw.rect(surface, WHITE, (ROAD_LEFT, 0, 4, WINDOW_HEIGHT))
    pygame.draw.rect(surface, WHITE, (ROAD_RIGHT - 4, 0, 4, WINDOW_HEIGHT))


def draw_text(
    surface: pygame.Surface,
    text: str,
    size: int,
    color: tuple[int, int, int],
    center: tuple[int, int],
) -> None:
    font = pygame.font.SysFont("arial", size, bold=True)
    rendered = font.render(text, True, color)
    surface.blit(rendered, rendered.get_rect(center=center))


def spawn_traffic_car() -> Car:
    lane = random.randint(0, LANE_COUNT - 1)
    x = ROAD_LEFT + lane * LANE_WIDTH + LANE_WIDTH / 2
    color = random.choice([RED, BLUE, (230, 150, 40), (160, 90, 220)])
    return Car(x, -CAR_HEIGHT, color)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Virtual Steering Wheel - Racing Game")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    spawn_event = pygame.USEREVENT + 1
    pygame.time.set_timer(spawn_event, TRAFFIC_SPAWN_MS)

    def reset_game():
        player = Car(WINDOW_WIDTH / 2, WINDOW_HEIGHT - 140, (40, 200, 90))
        return {
            "player": player,
            "speed": 0.0,
            "scroll": 0.0,
            "traffic": [],
            "score": 0.0,
            "game_over": False,
        }

    state = reset_game()

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == spawn_event and not state["game_over"]:
                state["traffic"].append(spawn_traffic_car())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r and state["game_over"]:
                    state = reset_game()

        keys = pygame.key.get_pressed()
        # Read hand tracking control from the bridge file
        bridge = read_control()
        player: Car = state["player"]

        if not state["game_over"]:
            # Use bridge input if calibrated and recent, fall back to keyboard
            use_bridge = bridge.calibrated and (time.time() - bridge.timestamp < 0.5)

            # Accelerate / brake (W / S) -- from bridge or keyboard.
            if use_bridge:
                if bridge.accelerate:
                    state["speed"] = min(
                        PLAYER_MAX_SPEED, state["speed"] + PLAYER_ACCELERATION
                    )
                elif bridge.brake:
                    state["speed"] = max(0.0, state["speed"] - PLAYER_BRAKE_DECEL)
                else:
                    state["speed"] = max(0.0, state["speed"] - PLAYER_FRICTION)

                # Steer left / right from bridge.
                if bridge.steer_left:
                    player.x -= STEER_SPEED
                if bridge.steer_right:
                    player.x += STEER_SPEED
            else:
                # Keyboard fallback
                if keys[pygame.K_w]:
                    state["speed"] = min(
                        PLAYER_MAX_SPEED, state["speed"] + PLAYER_ACCELERATION
                    )
                elif keys[pygame.K_s]:
                    state["speed"] = max(0.0, state["speed"] - PLAYER_BRAKE_DECEL)
                else:
                    state["speed"] = max(0.0, state["speed"] - PLAYER_FRICTION)

                # Steer left / right (A / D).
                if keys[pygame.K_a]:
                    player.x -= STEER_SPEED
                if keys[pygame.K_d]:
                    player.x += STEER_SPEED

            half_car = CAR_WIDTH / 2
            player.x = max(
                ROAD_LEFT + half_car + 4, min(ROAD_RIGHT - half_car - 4, player.x)
            )
            player.update_rect()

            state["scroll"] += state["speed"]
            state["score"] += state["speed"] * 0.05

            for car in state["traffic"]:
                car.y += state["speed"] * 0.5 + TRAFFIC_MIN_SPEED
                car.update_rect()
            state["traffic"] = [
                c for c in state["traffic"] if c.y < WINDOW_HEIGHT + CAR_HEIGHT
            ]

            for car in state["traffic"]:
                if player.rect.colliderect(car.rect):
                    state["game_over"] = True

        draw_road(screen, state["scroll"])
        for car in state["traffic"]:
            car.draw(screen)
        player.draw(screen)

        draw_text(
            screen, f"Score: {int(state['score'])}", 28, WHITE, (90, 30)
        )
        draw_text(
            screen, f"Speed: {state['speed']:.1f}", 20, WHITE, (90, 60)
        )

        if state["game_over"]:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(160)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))
            draw_text(
                screen, "GAME OVER", 48, RED,
                (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30),
            )
            draw_text(
                screen, "Press R to restart", 24, WHITE,
                (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20),
            )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

