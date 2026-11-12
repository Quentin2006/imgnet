import time
from pathlib import Path

import numpy as np


class NeuralNet:
    def __init__(
        self, inputDim: int, outputDim: int, hiddenDim: int, numHidden: int
    ) -> None:
        """
        we can encode nerual nets by using matricies.

        from one layer to another, we can define the activation of the next layer nerons by

        sig(weight_mat * activation_vec + bias_vec) = next_layer_activations
        """
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hiddenDim = hiddenDim
        self.numHidden = numHidden

        rng = np.random.default_rng()
        sizes = [inputDim] + [hiddenDim] * numHidden + [outputDim]

        # weights[l] maps layer l (n_in neurons) -> layer l + 1 (n_out neurons).
        # Start at normal(0, 1/sqrt(n_in)) so each neuron's z starts small; a flat
        # uniform(-1, 1) saturates the sigmoid in wide layers and they never learn.
        self.weights = [
            rng.normal(0, 1 / np.sqrt(n_in), (n_out, n_in))
            for n_in, n_out in zip(sizes, sizes[1:])
        ]
        self.biases = [np.zeros(n_out) for n_out in sizes[1:]]

    def sig(self, z: np.ndarray) -> np.ndarray:
        # clip so exp() can't overflow (sig is already 0/1 out there)
        return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

    def sig_prime(self, z: np.ndarray) -> np.ndarray:
        s = self.sig(z)
        return s * (1 - s)

    def forward(self, input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Run the net on a single (inputDim,) vector or a (batch, inputDim) matrix.

        Returns every layer's activations (input first, output last) and every
        weighted input z, where zs[l] produced activations[l + 1].
        """
        activations = [np.asarray(input)]
        zs = []
        for w, b in zip(self.weights, self.biases):
            # a @ W.T handles a single vector or a batch, one row per sample
            z = activations[-1] @ w.T + b
            zs.append(z)
            activations.append(self.sig(z))
        return activations, zs

    def _one_hot(self, targets) -> np.ndarray:
        targets = np.asarray(targets).reshape(-1)
        y = np.zeros((targets.shape[0], self.outputDim))
        y[np.arange(targets.shape[0]), targets] = 1.0
        return y

    def accuracy(self, inputs, targets) -> float:
        """Fraction of samples whose highest output matches the target."""
        activations, _ = self.forward(np.asarray(inputs))
        return float((activations[-1].argmax(axis=1) == np.asarray(targets)).mean())

    def backprop(
        self, activations: list[np.ndarray], zs: list[np.ndarray], targets
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Returns gradients shaped like self.weights and self.biases, averaged
        # over the batch. Every array here has one row per sample in the batch.

        # One-hot desired outputs, (batch, outputDim).
        y = self._one_hot(targets)
        batch = y.shape[0]

        step_w = [None] * len(self.weights)
        step_b = [None] * len(self.biases)

        # dC/da for the output layer.
        dC_da = 2 * (activations[-1] - y)

        # Walk backwards through the layers.
        for l in reversed(range(len(self.weights))):
            # dC/dz_j = sig'(z_j) * dC/da_j
            delta = self.sig_prime(zs[l]) * dC_da

            # dC/dw_jk = a_k * dC/dz_j, summed over the batch then averaged
            step_w[l] = (delta.T @ activations[l]) / batch
            # dC/db_j = dC/dz_j
            step_b[l] = delta.mean(axis=0)

            # dC/da_k for the previous layer = sum_j w_jk * dC/dz_j
            dC_da = delta @ self.weights[l]

        return step_w, step_b

    def train(
        self,
        inputs,
        targets,
        epochs: int,
        learning_rate: float = 3.0,
        batch_size: int = 32,
    ) -> None:
        # Mini-batch gradient descent: each step runs batch_size samples through
        # the net as one matrix, averages their gradients and takes one step
        # downhill by subtracting them from the weights.
        X = np.asarray(inputs)
        T = np.asarray(targets)
        n = X.shape[0]

        start = time.perf_counter()
        print(
            f"[train] starting: {epochs} epochs x {n} samples "
            f"(batch {batch_size}, lr {learning_rate})",
            flush=True,
        )

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            # shuffle each epoch so batches differ
            order = np.random.permutation(n)

            for s in range(0, n, batch_size):
                idx = order[s : s + batch_size]
                activations, zs = self.forward(X[idx])
                step_w, step_b = self.backprop(activations, zs, T[idx])

                for l in range(len(self.weights)):
                    self.weights[l] -= learning_rate * step_w[l]
                    self.biases[l] -= learning_rate * step_b[l]

            total = time.perf_counter() - start
            eta = total / (epoch + 1) * (epochs - epoch - 1)
            print(
                f"[train] epoch {epoch + 1}/{epochs} done in "
                f"{time.perf_counter() - epoch_start:.1f}s "
                f"(total {total:.1f}s, ETA {eta:.1f}s)",
                flush=True,
            )

        print(f"[train] finished in {time.perf_counter() - start:.1f}s", flush=True)

    def save(self, path: str | Path) -> None:
        """Write shape info, weights and biases to a single .npz file."""
        arrays = {
            "dims": np.array(
                [self.inputDim, self.outputDim, self.hiddenDim, self.numHidden]
            )
        }
        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{l}"] = w
            arrays[f"b{l}"] = b
        np.savez(path, **arrays)
        print(f"[save] wrote weights to {path}", flush=True)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNet":
        """Rebuild a network from a file written by save()."""
        data = np.load(path)
        net = cls(*(int(v) for v in data["dims"]))
        net.weights = [data[f"w{l}"] for l in range(len(net.weights))]
        net.biases = [data[f"b{l}"] for l in range(len(net.biases))]
        print(f"[load] loaded weights from {path}", flush=True)
        return net
