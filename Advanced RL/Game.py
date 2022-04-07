import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)


# font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:

    def __init__(self, width=640, height=480, block_size=20, game_speed=50, window_title="RL Snake"):
        self._width = width
        self._height = height
        self._block_size = block_size
        self._game_speed = game_speed
        # init display
        self.display = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(window_title)
        self.clock = pygame.time.Clock()

        self.direction = None
        self.snake_head = None
        self.snake_body = None
        self.rat = None
        self.score = 0
        self.frame_iteration = 0
        self.reset()

    # Getter and setter
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        if width > 0:
            self._width = width

    @property
    def height(self):
        return self._height

    @height.setter
    def width(self, height):
        if height > 0:
            self._height = height

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, block_size):
        if block_size > 0:
            self._block_size = block_size

    @property
    def game_speed(self):
        return self._game_speed

    @game_speed.setter
    def game_speed(self, game_speed):
        if game_speed > 0:
            self._game_speed = game_speed

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.snake_head = Point(self._width / 2, self._height / 2)
        self.snake_body = [self.snake_head,
                           Point(self.snake_head.x - self._block_size, self.snake_head.y),
                           Point(self.snake_head.x - (2 * self._block_size), self.snake_head.y)]

        self.score = 0
        self.rat = None
        self._place_rat()
        self.frame_iteration = 0

    def _place_rat(self):
        x = random.randint(0, (self._width - self._block_size) // self._block_size) * self._block_size
        y = random.randint(0, (self._height - self._block_size) // self._block_size) * self._block_size
        self.rat = Point(x, y)
        if self.rat in self.snake_body:
            self._place_rat()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake_body.insert(0, self.snake_head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake_body):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.snake_head == self.rat:
            self.score += 1
            reward = 10
            self._place_rat()
        else:
            self.snake_body.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self._game_speed)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake_head
        # hits boundary
        if pt.x > self._width - self._block_size or pt.x < 0 or pt.y > self._height - self._block_size or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake_body[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake_body:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self._block_size, self._block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.rat.x, self.rat.y, self._block_size, self._block_size))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.snake_head.x
        y = self.snake_head.y
        if self.direction == Direction.RIGHT:
            x += self._block_size
        elif self.direction == Direction.LEFT:
            x -= self._block_size
        elif self.direction == Direction.DOWN:
            y += self._block_size
        elif self.direction == Direction.UP:
            y -= self._block_size

        self.snake_head = Point(x, y)
