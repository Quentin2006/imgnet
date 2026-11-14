# Draw a digit and watch the network classify it, layer by layer.
import sys
from pathlib import Path

import numpy as np
import pygame
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nn

# --- network settings ---
WEIGHTS_PATH = Path(__file__).with_name("weights.npz")
# 5 hidden layers of 144 (12x12 squares on screen) gets ~97.4% test accuracy.
# With this many sigmoid layers, wider ones (e.g. 256) stop learning because
# the gradient fades on its way back; 2 x 256 gets ~98.2% if depth isn't needed.
HIDDEN_DIM = 144
NUM_HIDDEN = 5
LEARNING_RATE = 1.0
# accuracy levels off around 50 epochs; past ~100 it just memorizes
EPOCHS = 50

# --- layout ---
INPUT_WIDTH = 28  # MNIST images are 28x28
CELL_SIZE = 25  # 1 dataset pixel == 25x25 screen pixels
GRID_PX = INPUT_WIDTH * CELL_SIZE
# each hidden layer is drawn as the smallest square grid that fits it
# (144 -> 12x12, 256 -> 16x16); any leftover cells stay empty
NEURON_PX = 16
LAYER_COLS = LAYER_ROWS = int(np.ceil(np.sqrt(HIDDEN_DIM)))
LAYER_W = LAYER_COLS * NEURON_PX
LAYER_GAP = 80  # space between layer grids for the connection lines
NET_X = GRID_PX  # network view, right of the drawing
NET_W = 50 + NUM_HIDDEN * LAYER_W + (NUM_HIDDEN - 1) * LAYER_GAP + 40
BARS_X, BARS_W = NET_X + NET_W, 280  # prediction bars, far right
LAYER_X = [NET_X + 50 + k * (LAYER_W + LAYER_GAP) for k in range(NUM_HIDDEN)]
LAYER_Y = (GRID_PX - LAYER_ROWS * NEURON_PX) // 2
TOP_LINES = 30  # strongest connections drawn per layer pair
ROW_H = 52  # prediction bar rows

BG = (0, 0, 0)
PANEL = (30, 30, 36)
DIM_TEXT = (140, 140, 150)
GREEN = (120, 220, 140)
RED = (230, 90, 90)


def load_mnist(split: str):
    ds = load_dataset("ylecun/mnist", split=split)
    data = ds.with_format("numpy")[:]  # whole split as numpy arrays
    images = data["image"].reshape(len(ds), -1).astype(np.float32) / 255.0
    return images, data["label"]


def get_network() -> nn.NeuralNet:
    # reuse saved weights unless asked to retrain (python main.py --retrain)
    if WEIGHTS_PATH.exists() and "--retrain" not in sys.argv:
        return nn.NeuralNet.load(WEIGHTS_PATH)

    network = nn.NeuralNet(INPUT_WIDTH * INPUT_WIDTH, 10, HIDDEN_DIM, NUM_HIDDEN)
    network.train(*load_mnist("train"), EPOCHS, learning_rate=LEARNING_RATE)
    network.save(WEIGHTS_PATH)
    acc = network.accuracy(*load_mnist("test"))
    print(f"[eval] test accuracy: {acc * 100:.2f}%", flush=True)
    return network


def handle_input(grid: np.ndarray) -> None:
    """Left click draws, F erases, C clears. grid is indexed [row][col]."""
    mouse_x, mouse_y = pygame.mouse.get_pos()
    row, col = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
    on_grid = 0 <= row < INPUT_WIDTH and 0 <= col < INPUT_WIDTH
    keys = pygame.key.get_pressed()

    if on_grid and pygame.mouse.get_pressed()[0]:
        grid[row][col] = 1
    if on_grid and keys[pygame.K_f]:
        grid[row][col] = 0
    if keys[pygame.K_c]:
        grid.fill(0)


# --- where each neuron sits on screen, for drawing connection lines ---


def input_pos(i: int) -> tuple[int, int]:
    row, col = divmod(i, INPUT_WIDTH)
    return col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2


def hidden_pos(layer: int, i: int) -> tuple[int, int]:
    row, col = divmod(i, LAYER_COLS)
    return (
        LAYER_X[layer] + col * NEURON_PX + NEURON_PX // 2,
        LAYER_Y + row * NEURON_PX + NEURON_PX // 2,
    )


def output_pos(digit: int) -> tuple[int, int]:
    # left edge of that digit's row in the prediction panel
    return BARS_X + 10, 80 + digit * ROW_H + 13


def neuron_pos(layer: int, i: int) -> tuple[int, int]:
    # layer 0 = input pixels, 1..NUM_HIDDEN = hidden layers, last = output
    if layer == 0:
        return input_pos(i)
    if layer <= NUM_HIDDEN:
        return hidden_pos(layer - 1, i)
    return output_pos(i)


def draw_drawing(screen, grid: np.ndarray) -> None:
    for row, col in zip(*np.nonzero(grid)):
        rect = (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (255, 255, 255), rect)


def draw_network(screen, font, activations, weights) -> None:
    """Hidden neurons shaded by activation, plus the strongest live connections."""
    pygame.draw.rect(screen, (18, 18, 24), (NET_X, 0, NET_W, GRID_PX))

    # Lines: a connection's push on the next neuron is weight * source activation.
    # Green pushes the next neuron up, red pushes it down; brighter = stronger.
    for layer, (w, a) in enumerate(zip(weights, activations)):
        contrib = w * a  # (n_out, n_in)
        flat = np.abs(contrib).ravel()
        top = np.argpartition(flat, -TOP_LINES)[-TOP_LINES:]
        strongest = flat[top].max()
        if strongest == 0:
            continue  # nothing drawn yet
        for k in top:
            dst, src = divmod(int(k), contrib.shape[1])
            strength = 0.25 + 0.75 * flat[k] / strongest
            base = GREEN if contrib[dst, src] > 0 else RED
            color = tuple(int(c * strength) for c in base)
            start, end = neuron_pos(layer, src), neuron_pos(layer + 1, dst)
            pygame.draw.line(screen, color, start, end, 1)

    # Neurons: dark at activation 0, bright yellow at 1.
    for layer, acts in enumerate(activations[1:-1]):
        label = f"Hidden {layer + 1} ({len(acts)})"
        # solid background so connection lines don't run through the text
        text = font.render(label, True, DIM_TEXT, (18, 18, 24))
        screen.blit(text, (LAYER_X[layer], LAYER_Y - 28))
        for i, v in enumerate(acts):
            x, y = hidden_pos(layer, i)
            color = (int(25 + 230 * v), int(25 + 175 * v), int(35 + 25 * v))
            half = NEURON_PX // 2
            pygame.draw.rect(screen, color, (x - half, y - half, NEURON_PX - 1, NEURON_PX - 1))


def draw_predictions(screen, fonts, outputs) -> None:
    title_font, row_font, hint_font = fonts
    # sigmoid outputs are independent, so normalize them into percentages
    total = outputs.sum()
    probs = outputs / total if total > 0 else np.full(10, 0.1)
    best = int(np.argmax(probs))

    pygame.draw.rect(screen, PANEL, (BARS_X, 0, BARS_W, GRID_PX))
    px = BARS_X + 20
    screen.blit(title_font.render(f"Guess: {best}", True, GREEN), (px, 20))

    bar_x = px + 30
    bar_max = BARS_W - 40 - 30 - 70  # room for the digit label and % text
    for d in range(10):
        y = 80 + d * ROW_H
        color = GREEN if d == best else (110, 150, 220)
        text_color = (255, 255, 255) if d == best else (190, 190, 200)

        screen.blit(row_font.render(str(d), True, text_color), (px, y + 4))
        pygame.draw.rect(screen, (55, 55, 65), (bar_x, y, bar_max, 26), border_radius=4)
        fill = int(bar_max * probs[d])
        if fill > 0:
            pygame.draw.rect(screen, color, (bar_x, y, fill, 26), border_radius=4)
        pct = row_font.render(f"{probs[d] * 100:5.1f}%", True, text_color)
        screen.blit(pct, (bar_x + bar_max + 8, y + 4))

    hint = "click: draw   F: erase   C: clear"
    screen.blit(hint_font.render(hint, True, DIM_TEXT), (px, GRID_PX - 30))


def main() -> None:
    network = get_network()

    pygame.init()
    screen = pygame.display.set_mode((BARS_X + BARS_W, GRID_PX))
    pygame.display.set_caption("Digit recognizer")
    pygame.mouse.set_visible(False)
    fonts = [pygame.font.SysFont(None, size) for size in (40, 30, 22)]
    clock = pygame.time.Clock()

    # the drawing, indexed [row][col] like MNIST: 1 = white, 0 = black
    grid = np.zeros((INPUT_WIDTH, INPUT_WIDTH))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        handle_input(grid)

        activations, _ = network.forward(grid.flatten())

        screen.fill(BG)
        draw_drawing(screen, grid)
        draw_predictions(screen, fonts, activations[-1])
        draw_network(screen, fonts[2], activations, network.weights)
        # cursor last so it draws over everything
        pygame.draw.circle(screen, (255, 0, 0), pygame.mouse.get_pos(), 7)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
