# Example file showing a basic pygame "game loop"
import sys
from pathlib import Path

import numpy as np
import pygame
from datasets import load_dataset

import nn

# pygame setup
pygame.init()

CELL_SIZE = 25
INPUT_WIDTH = 28
GRID_PX = INPUT_WIDTH * CELL_SIZE
PANEL_WIDTH = 280
WEIGHTS_PATH = Path(__file__).with_name("weights.npz")
# 2 hidden layers of 256 was the best size tested (~98.2% validation accuracy);
# wider layers were no better and need a smaller learning rate.
HIDDEN_DIM = 256
NUM_HIDDEN = 2
LEARNING_RATE = 1.0
# accuracy levels off around 50 epochs; past ~100 it just memorizes
EPOCHS = 50


def load_mnist(split: str):
    ds = load_dataset("ylecun/mnist", split=split)
    data = ds.with_format("numpy")[:]  # whole split as numpy arrays
    images = data["image"].reshape(len(ds), -1).astype(np.float32) / 255.0
    return images, data["label"]


# reuse saved weights unless asked to retrain (python main.py --retrain)
if WEIGHTS_PATH.exists() and "--retrain" not in sys.argv:
    network = nn.NeuralNet.load(WEIGHTS_PATH)
else:
    network = nn.NeuralNet(INPUT_WIDTH * INPUT_WIDTH, 10, HIDDEN_DIM, NUM_HIDDEN)
    inputs, targets = load_mnist("train")
    network.train(inputs, targets, EPOCHS, learning_rate=LEARNING_RATE)
    network.save(WEIGHTS_PATH)
    test_inputs, test_targets = load_mnist("test")
    acc = network.accuracy(test_inputs, test_targets)
    print(f"[eval] test accuracy: {acc * 100:.2f}%", flush=True)

# dataset has 28x28 images, so we can make 1 pixel in the dataset == 25 real pixels
# extra space on the right holds the prediction panel
screen = pygame.display.set_mode((GRID_PX + PANEL_WIDTH, GRID_PX))
pygame.display.set_caption("Digit recognizer")
title_font = pygame.font.SysFont(None, 40)
row_font = pygame.font.SysFont(None, 30)
hint_font = pygame.font.SysFont(None, 22)
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

    # only edit the grid while the cursor is over it (not the side panel)
    i, j = mouse_x // CELL_SIZE, mouse_y // CELL_SIZE
    on_grid = 0 <= i < INPUT_WIDTH and 0 <= j < INPUT_WIDTH

    # left click
    if on_grid and pygame.mouse.get_pressed()[0]:
        userVals[i][j] = True
    if on_grid and pygame.key.get_pressed()[pygame.K_f]:
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
    # userVals is indexed [x][y]; MNIST flattens row-major [y][x], so transpose first
    outputs = network.forward(userVals.T.flatten())[0][-1]
    # sigmoid outputs are independent, so normalize them into percentages
    total = outputs.sum()
    probs = outputs / total if total > 0 else np.full(10, 0.1)
    best = int(np.argmax(probs))

    # prediction panel
    pygame.draw.rect(screen, (30, 30, 36), (GRID_PX, 0, PANEL_WIDTH, GRID_PX))
    pygame.draw.line(screen, (80, 80, 90), (GRID_PX, 0), (GRID_PX, GRID_PX), 2)
    px = GRID_PX + 20
    screen.blit(title_font.render(f"Guess: {best}", True, (120, 220, 140)), (px, 20))

    bar_max = PANEL_WIDTH - 40 - 30 - 70  # room for the digit label and % text
    row_h = 52
    for d in range(10):
        y = 80 + d * row_h
        is_best = d == best
        color = (120, 220, 140) if is_best else (110, 150, 220)
        text_color = (255, 255, 255) if is_best else (190, 190, 200)

        screen.blit(row_font.render(str(d), True, text_color), (px, y + 4))
        bar_x = px + 30
        pygame.draw.rect(screen, (55, 55, 65), (bar_x, y, bar_max, 26), border_radius=4)
        fill = int(bar_max * probs[d])
        if fill > 0:
            pygame.draw.rect(screen, color, (bar_x, y, fill, 26), border_radius=4)
        screen.blit(
            row_font.render(f"{probs[d] * 100:5.1f}%", True, text_color),
            (bar_x + bar_max + 8, y + 4),
        )

    hint = "click: draw   F: erase   C: clear"
    screen.blit(hint_font.render(hint, True, (140, 140, 150)), (px, GRID_PX - 30))

    # cursor last so it draws over everything
    pygame.draw.circle(screen, (255, 0, 0), (mouse_x, mouse_y), 7)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(175)

pygame.quit()
