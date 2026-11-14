# Type text and watch a character-level network predict what comes next.
import os
import sys
import time
import urllib.request
from pathlib import Path

# Small matrices run much faster without a large BLAS thread pool.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nn

# --- network settings ---
CONTEXT = 3  # characters the net sees; 1 = bigram model
LOWERCASE = True
HIDDEN_DIM = 256
NUM_HIDDEN = 2
EPOCHS = 3
LEARNING_RATE = 4.0
BATCH_SIZE = 256
TEMPERATURE = 0.8
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = Path(__file__).with_name('input.txt')
WEIGHTS_PATH = Path(__file__).with_name('weights.npz')

# --- layout ---
TOP_LINES = 30
BG = (0, 0, 0)
PANEL = (30, 30, 36)
DIM_TEXT = (140, 140, 150)
GREEN = (120, 220, 140)
RED = (230, 90, 90)
NEURON_PX = 16
LAYER_GAP = 80
BARS_W = 300
ROW_H = 45


def load_text():
    if not DATA_PATH.exists():
        print(f'[data] downloading {DATA_URL}', flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            DATA_PATH.write_bytes(response.read())
    text = DATA_PATH.read_text(encoding='utf-8')
    return text.lower() if LOWERCASE else text


def encode(indices, vocab_size):
    indices = np.asarray(indices)
    out = np.zeros((len(indices), CONTEXT, vocab_size), dtype=np.float32)
    out[np.arange(len(indices))[:, None], np.arange(CONTEXT), indices] = 1
    return out.reshape(len(indices), -1)


def input_vector(text, vocab):
    lookup = {c: i for i, c in enumerate(vocab)}
    pad = ' ' if ' ' in lookup else vocab[0]
    context = (pad * CONTEXT + text)[-CONTEXT:]
    return encode([[lookup.get(c, lookup[pad]) for c in context]], len(vocab))[0]


def sample_char(network, text, vocab):
    probs = network.forward(input_vector(text, vocab))[0][-1]
    logits = np.log(np.clip(probs, 1e-12, 1)) / TEMPERATURE
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return vocab[int(np.random.choice(len(vocab), p=probs))]


def get_network(text, vocab):
    if WEIGHTS_PATH.exists() and '--retrain' not in sys.argv:
        network = nn.NeuralNet.load(WEIGHTS_PATH)
        if (network.inputDim, network.outputDim, network.output) == (
            CONTEXT * len(vocab), len(vocab), 'softmax'
        ):
            return network
        print('[train] saved dimensions or output mode differ; retraining', flush=True)
    else:
        print('[train] no saved weights or --retrain requested', flush=True)

    lookup = {c: i for i, c in enumerate(vocab)}
    chars = np.fromiter((lookup[c] for c in text), dtype=np.int32, count=len(text))
    windows = np.lib.stride_tricks.sliding_window_view(chars, CONTEXT + 1)
    split = int(len(windows) * 0.9)
    network = nn.NeuralNet(CONTEXT * len(vocab), len(vocab), HIDDEN_DIM, NUM_HIDDEN, output='softmax')
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        order = np.random.permutation(split)
        for s in range(0, split, BATCH_SIZE):
            batch = windows[order[s:s + BATCH_SIZE]]
            acts, zs = network.forward(encode(batch[:, :-1], len(vocab)))
            dw, db = network.backprop(acts, zs, batch[:, -1])
            for layer in range(len(network.weights)):
                network.weights[layer] -= LEARNING_RATE * dw[layer]
                network.biases[layer] -= LEARNING_RATE * db[layer]
        print(f'[train] epoch {epoch + 1}/{EPOCHS}, {time.perf_counter() - start:.1f}s', flush=True)
    print(f'[train] finished in {time.perf_counter() - start:.1f}s', flush=True)
    network.save(WEIGHTS_PATH)
    loss = correct = count = 0
    for s in range(split, len(windows), 2048):
        batch = windows[s:s + 2048]
        x, y = encode(batch[:, :-1], len(vocab)), batch[:, -1]
        loss += network.loss(x, y) * len(y)
        correct += network.accuracy(x, y) * len(y)
        count += len(y)
    print(f'[eval] validation loss: {loss / count:.4f}; next-char accuracy: {100 * correct / count:.2f}%', flush=True)
    print(f'[eval] parameters: {sum(w.size + b.size for w, b in zip(network.weights, network.biases))}', flush=True)
    generated = 'the '
    for _ in range(300):
        generated += sample_char(network, generated, vocab)
    print(f'[sample] {generated}', flush=True)
    return network


def label(c):
    return {' ': 'sp', '\n': 'nl', '\t': 'tab'}.get(c, c)


def layout(vocab, network):
    row = min(20, max(8, 760 // len(vocab)))
    cell = min(26, max(12, 220 // CONTEXT))
    input_w = 55 + CONTEXT * cell
    side = int(np.ceil(np.sqrt(network.hiddenDim)))
    layer_w = side * NEURON_PX
    layer_x = [input_w + 70 + k * (layer_w + LAYER_GAP) for k in range(network.numHidden)]
    bars_x = input_w + 70 + network.numHidden * (layer_w + LAYER_GAP)
    height = max(600, 70 + len(vocab) * row)
    return row, cell, input_w, side, layer_x, bars_x, height


def draw_frame(screen, fonts, text, vocab, network):
    title_font, row_font, hint_font = fonts
    row, cell, input_w, side, layer_x, bars_x, height = layout(vocab, network)
    acts, _ = network.forward(input_vector(text, vocab))
    outputs = acts[-1]
    top_outputs = np.argsort(outputs)[-10:][::-1]
    ranks = {int(v): i for i, v in enumerate(top_outputs)}
    layer_y = (height - side * NEURON_PX) // 2
    screen.fill(BG)
    pygame.draw.rect(screen, (18, 18, 24), (input_w, 0, bars_x - input_w, height))
    pygame.draw.rect(screen, PANEL, (bars_x, 0, BARS_W, height))

    def pos(layer, i):
        if layer == 0:
            col, r = divmod(i, len(vocab))
            return 45 + col * cell + cell // 2, 48 + r * row + row // 2
        if layer <= network.numHidden:
            r, col = divmod(i, side)
            return layer_x[layer - 1] + col * NEURON_PX + NEURON_PX // 2, layer_y + r * NEURON_PX + NEURON_PX // 2
        return bars_x + 10, 84 + ranks.get(i, 0) * ROW_H

    # Lines show weight * source activation; green pushes up and red down.
    for layer, (w, a) in enumerate(zip(network.weights, acts)):
        contrib = w * a
        flat = np.abs(contrib).ravel()
        top = np.argpartition(flat, -min(TOP_LINES, flat.size))[-min(TOP_LINES, flat.size):]
        strongest = flat[top].max()
        if strongest == 0:
            continue
        for k in top:
            dst, src = divmod(int(k), contrib.shape[1])
            if layer == len(network.weights) - 1 and dst not in ranks:
                continue
            strength = 0.25 + 0.75 * flat[k] / strongest
            base = GREEN if contrib[dst, src] > 0 else RED
            pygame.draw.line(screen, tuple(int(c * strength) for c in base), pos(layer, src), pos(layer + 1, dst))

    for col in range(CONTEXT):
        screen.blit(hint_font.render(f't-{CONTEXT-col}', True, DIM_TEXT), (45 + col * cell, 23))
    for r, char in enumerate(vocab):
        y = 48 + r * row
        screen.blit(hint_font.render(label(char), True, DIM_TEXT), (8, y))
        for col in range(CONTEXT):
            active = acts[0][col * len(vocab) + r]
            color = (240, 225, 130) if active else (45, 45, 52)
            pygame.draw.rect(screen, color, (45 + col * cell, y, cell - 2, row - 2))
    for layer, values in enumerate(acts[1:-1]):
        screen.blit(hint_font.render(f'Hidden {layer + 1} ({len(values)})', True, DIM_TEXT, (18, 18, 24)), (layer_x[layer], layer_y - 25))
        for i, value in enumerate(values):
            x, y = pos(layer + 1, i)
            color = (int(25 + 230 * value), int(25 + 175 * value), int(35 + 25 * value))
            pygame.draw.rect(screen, color, (x - 8, y - 8, NEURON_PX - 1, NEURON_PX - 1))
    px = bars_x + 20
    screen.blit(title_font.render(f'Next: {label(vocab[int(top_outputs[0])])}', True, GREEN), (px, 20))
    for rank, idx in enumerate(top_outputs):
        y = 72 + rank * ROW_H
        color = GREEN if rank == 0 else (110, 150, 220)
        screen.blit(row_font.render(label(vocab[int(idx)]), True, (230, 230, 235)), (px, y))
        pygame.draw.rect(screen, (55, 55, 65), (px + 35, y, 155, 23), border_radius=4)
        pygame.draw.rect(screen, color, (px + 35, y, int(155 * outputs[idx]), 23), border_radius=4)
        screen.blit(hint_font.render(f'{outputs[idx] * 100:4.1f}%', True, (220, 220, 225)), (px + 198, y + 3))
    screen.blit(hint_font.render('type: add   backspace: delete', True, DIM_TEXT), (px, height - 60))
    screen.blit(hint_font.render('tab (hold): generate   esc: clear', True, DIM_TEXT), (px, height - 38))
    # A bottom strip keeps the latest text visible, with the context highlighted.
    strip_y = height
    pygame.draw.rect(screen, PANEL, (0, strip_y, screen.get_width(), 70))
    shown = text.replace('\n', '↵')[-max(1, screen.get_width() // 4):]
    while shown and row_font.size(shown)[0] > screen.get_width() - 30:
        shown = shown[1:]
    x = 15
    for i, char in enumerate(shown):
        color = (245, 220, 120) if i >= len(shown) - CONTEXT else (210, 210, 220)
        image = row_font.render(char, True, color)
        screen.blit(image, (x, strip_y + 22))
        x += image.get_width()


def main():
    text_data = load_text()
    vocab = sorted(set(text_data))
    network = get_network(text_data, vocab)
    pygame.init()
    *_, bars_x, height = layout(vocab, network)
    screen = pygame.display.set_mode((bars_x + BARS_W, height + 70))
    pygame.display.set_caption('Text predictor')
    fonts = [pygame.font.SysFont(None, size) for size in (40, 26, 20)]
    clock = pygame.time.Clock()
    typed = 'the '
    last_sample = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.TEXTINPUT:
                for char in event.text.lower() if LOWERCASE else event.text:
                    if char in vocab:
                        typed += char
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    typed = typed[:-1]
                elif event.key == pygame.K_ESCAPE:
                    typed = ''
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and '\n' in vocab:
                    typed += '\n'
        now = pygame.time.get_ticks()
        if pygame.key.get_pressed()[pygame.K_TAB] and now - last_sample >= 100:
            typed += sample_char(network, typed, vocab)
            last_sample = now
        draw_frame(screen, fonts, typed, vocab, network)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == '__main__':
    main()
