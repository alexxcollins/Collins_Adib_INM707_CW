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

BLOCK = (166, 166, 166)

GREEN = (0, 153, 51)
GREEN1 = (153, 255, 153)
GREEN2 = (102, 255, 51)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:

    def __init__(self,
                 width=640,
                 height=480,
                 window_title="Reinforcement Learning Snake",
                 block_type = ['line', 'square'],
                 block_size=20,
                 game_speed=40):
        """
        Constructor for the SnakeGame
        :param width (int64): width of the window that will appear on the screen
        :param height (int64): height of the window that will appear on the screen
        :param window_title (str): the title of the window that will appear on the screen
        :param block_type (list): list contains the types of blocks in the game
        :param block_size (int): the block size of each object in the game
        :param game_speed (int): the speed of the game (frame per second)
        """
        self.width = width
        self.height = height
        self.window_title = window_title
        self.block_type =  block_type
        self.block_size = block_size
        self.frame_iteration = None
        self.rat = None
        self.score = None
        self.snake_body = None
        self.snake_head = None
        self.blocks = []
        self.direction = None

        self.game_speed = game_speed
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.window_title)
        self.clock = pygame.time.Clock()
        self.reset()
        self._update_ui()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT


        self.snake_head = Point(self.width / 2, self.height / 2)
        self.snake_body = [self.snake_head,
                           Point(self.snake_head.x - self.block_size, self.snake_head.y),
                           Point(self.snake_head.x - (2 * self.block_size), self.snake_head.y)]

        for i in range(5):
            ok = False
            while not ok:
                x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
                y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
                b = Point(x, y)
                if b not in self.snake_body:
                    self.blocks.append(b)
                    ok = True

        # TODO:  add blocks randomly  in the game
        self.score = 0
        self.rat = None
        self._place_rat()
        self.frame_iteration = 0

    def _place_rat(self):
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.rat = Point(x, y)
        if self.rat in self.snake_body:
            self._place_rat()
        if self.rat in self.blocks:
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
            self.blocks = []
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.snake_head == self.rat:
            self.blocks = []
            self.score += 1
            reward = 10
            self._place_rat()
        else:
            self.snake_body.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.game_speed)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake_head
        # hits boundary
        if pt.x > self.width - self.block_size or pt.x < 0 or pt.y > self.height - self.block_size or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake_body[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for block in self.blocks:
            pygame.draw.rect(self.display, BLOCK, pygame.Rect(block.x, block.y, self.block_size, self.block_size))

        for pt in self.snake_body:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            #pygame.draw.circle(self.display, GREEN1, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            pygame.draw.circle(self.display, GREEN1, (pt.x+self.block_size/2, pt.y+self.block_size/2), 5, 0)

        pygame.draw.rect(self.display, GREEN2,
                         pygame.Rect(self.snake_head.x, self.snake_head.y, self.block_size, self.block_size))


        pygame.draw.rect(self.display, RED, pygame.Rect(self.rat.x, self.rat.y, self.block_size, self.block_size))

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
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.snake_head = Point(x, y)

    def get_observation(self, obs_dim=1):
        point_l = Point(self.snake_head.x - self.block_size * obs_dim, self.snake_head.y)
        point_r = Point(self.snake_head.x + self.block_size * obs_dim, self.snake_head.y)
        point_u = Point(self.snake_head.x, self.snake_head.y - self.block_size * obs_dim)
        point_d = Point(self.snake_head.x, self.snake_head.y + self.block_size * obs_dim)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.rat.x < self.snake_head.x,  # food left
            self.rat.x > self.snake_head.x,  # food right
            self.rat.y < self.snake_head.y,  # food up
            self.rat.y > self.snake_head.y  # food down
        ]

        return np.array(state, dtype=int)
