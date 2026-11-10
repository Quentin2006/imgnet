# Example file showing a basic pygame "game loop"
import numpy as np
import pygame

import nn

# pygame setup
pygame.init()

CELL_SIZE = 25
INPUT_WIDTH = 28


network = nn.NeuralNet(INPUT_WIDTH * INPUT_WIDTH, 10, 16, 2)

# dataset has 28x28 images, so we can make 1 pixel in the dataset == 25 real pixels
screen = pygame.display.set_mode((INPUT_WIDTH * CELL_SIZE, INPUT_WIDTH * CELL_SIZE))
clock = pygame.time.Clock()
running = True
pygame.mouse.set_visible(False)
# the dataset is a 28x28 matrix holding bools (1 for white, 0 for blacks)
userVals = np.zeros((INPUT_WIDTH, INPUT_WIDTH))

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    mouse_x, mouse_y = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # left click
    if pygame.mouse.get_pressed()[0]:
        i, j = mouse_x // CELL_SIZE, mouse_y // CELL_SIZE
        userVals[i][j] = True
    if pygame.key.get_pressed()[pygame.K_f]:
        i, j = mouse_x // CELL_SIZE, mouse_y // CELL_SIZE
        userVals[i][j] = False
    if pygame.key.get_pressed()[pygame.K_c]:
        userVals.fill(0)

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    # RENDER YOUR GAME HERE
    for x in range(INPUT_WIDTH):
        for y in range(INPUT_WIDTH):
            if userVals[x][y]:
                pygame.draw.rect(
                    screen,
                    (255, 255, 255),
                    (x * CELL_SIZE, y * CELL_SIZE, INPUT_WIDTH, INPUT_WIDTH),
                )
    pygame.draw.circle(screen, (255, 0, 0), (mouse_x, mouse_y), 7)
    # flip() the display to put your work on screen
    pygame.display.flip()

    print(network._cost(list(userVals.T.flatten()), 3))

    clock.tick(175)

pygame.quit()
