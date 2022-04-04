# import libraries
import pygame
import random
from collections import namedtuple
import numpy as np

# initialize pygame
pygame.init()

direction = {
    'RIGHT': 1,
    'LEFT': 2,
    'UP': 3,
    'DOWN': 4
}
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

font = pygame.font.Font('comicsansms', 25)


Point = namedtuple('Point', 'x', 'y')

# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


class Game:

    # constructor
    def __init__(self,
                 width=500,
                 height=500,
                 block_size=20,
                 speed=20,
                 window_title="Reinforcement Learning Snake"):
        """
        Constructor of the snake game
        :param width (int): width of the window
        :param height (int): height of the window
        :param block_size (int): size of the blocks on the screen
        :param speed (int): speed of the game
        :param window_title (str): Title of the window on the screen
        """
        self._width = width
        self._height = height
        self._block_size = block_size
        self._speed = speed
        self._window_title = window_title

        # Display the game
        self.display = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(self._window_title)

        self.clock = pygame.time.Clock()

        self.direction = None
        self.snake_head = None
        self.snake_body = None
        self.score = 0
        self.rat_position = None
        self.game_iteration = 0

        self.reset()

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
    def height(self, height):
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
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        if speed > 0:
            self._speed = speed

    @property
    def window_title(self):
        return self._window_title

    @window_title.setter
    def window_title(self, window_title):
        self._window_title = window_title

    # method to reset the environment
    def reset(self):
        # initialize the default game state
        self.direction = direction['RIGHT']
        self.snake_head = [self.width / 2, self._height / 2]
        self.snake_body = [self.snake_head,
                           [self.snake_head[0] - self.block_size, self.snake_head[1]],
                           [self.snake_head[0] - 2 * self.block_size, self.snake_head[1]]]

        self.score = 0
        self.game_iteration = 0
        self._place_rat()

    # Place the rate randomly on the map
    def _place_rat(self):
        x_rat = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y_rat = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.rat_position = [x_rat, y_rat]

        # to avoid initialize the rat position in the snake body
        if self.rat_position in self.snake_body:
            self._place_rat()

    # method to play a step in the game
    def play_step(self, action):
        # each time the agent take an action update the game iteration
        self.game_iteration += 1

        # quite the game manually
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move
        self._move(action)
        self.snake_body.insert(0, self.snake_head)

        # check game over
        reward = 0
        game_over = False
        # check if there is a collision or if the agent is doing random actions for long time
        if self._collision() or self.game_iteration > 500:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # place new rat
        if self.snake_head == self.rat_position:
            self._score += 1
            reward = 10
            self._place_rat()
        else:
            self.snake_body.pop()

        # update the user interface
        self._update_ui()
        self.clock.tick(self.speed)

        return reward, game_over, self.score

    # check collision
    def _collision(self, point=None):
        if point is None:
            point = self.snake_head

        # check if the snake hits the boundary
        if point[0] > self.width - self.block_size or point[0] < 0 or point[1] > self.height - self.block_size or point[1] < 0:
            return True

        # check if the snake hits itself
        if point in self.snake_body[1:]:
            return True

        return False

    # update the UI
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake_body:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.rat_position[0], self.rat_position[1], self.block_size, self.block_size))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # method to make the agent move
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [direction['RIGHT'], direction['DOWN'], direction['LEFT'], direction['UP']]
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

        x = self.snake_head[0]
        y = self.snake_head[1]
        if self.direction == direction['RIGHT']:
            x += self.block_size
        elif self.direction == direction['LEFT']:
            x -= self.block_size
        elif self.direction == direction['DOWN']:
            y += self.block_size
        elif self.direction == direction['UP']:
            y -= self.block_size

        self.head = [x, y]
