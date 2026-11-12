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

        rng = np.random.default_rng()

        # start weights at normal(0, 1/sqrt(fan_in)) so each neuron's z starts
        # small; a flat uniform(-1, 1) saturates the sigmoid in wide layers
        # and they never learn
        def init_w(rows: int, cols: int) -> np.ndarray:
            return rng.normal(0, 1 / np.sqrt(cols), (rows, cols))

        self.numLayers = 2 + numHidden
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.numHidden = numHidden
        self.hiddenDim = hiddenDim

        self.weights: list[np.ndarray] = []
        self.activations: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        # populate for input layer
        self.weights.append(init_w(hiddenDim, inputDim))
        self.activations.append(np.zeros(inputDim))
        self.biases.append(np.zeros(hiddenDim))

        # populate hiden layers
        for i in range(numHidden):
            self.activations.append(np.zeros(hiddenDim))
            if i < numHidden - 1:
                self.weights.append(init_w(hiddenDim, hiddenDim))
                self.biases.append(np.zeros(hiddenDim))

        # populate for output layer
        self.weights.append(init_w(outputDim, hiddenDim))
        self.activations.append(np.zeros(outputDim))
        self.biases.append(np.zeros(outputDim))

    def sig(self, z: np.ndarray) -> np.ndarray:
        # clip so exp() can't overflow (sig is already 0/1 out there)
        return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

    def sig_prime(self, z: np.ndarray) -> np.ndarray:
        s = self.sig(z)
        return s * (1 - s)

    def step(self, curLayer: int) -> np.ndarray:
        """
        will update curLayer+1 activations

        activations can be a single vector (n,) or a batch (batch, n);
        a @ W.T handles both, one row per sample.
        """
        curW, curA, curB = (
            self.weights[curLayer],
            self.activations[curLayer],
            self.biases[curLayer],
        )

        z = curA @ curW.T + curB

        self.activations[curLayer + 1] = self.sig(z)

        return z

    def forward(self, input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # seed input activations (a single (inputDim,) vector or a (batch, inputDim) matrix)
        self.activations[0] = np.asarray(input)

        zs = []

        for i in range(self.numLayers - 1):
            zs.append(self.step(i))

        return (self.activations, zs)

    def _one_hot(self, targets) -> np.ndarray:
        targets = np.asarray(targets).reshape(-1)
        y = np.zeros((targets.shape[0], self.outputDim))
        y[np.arange(targets.shape[0]), targets] = 1.0
        return y

    def cost(self, inputs, targets) -> float:
        # average squared error over every sample, done as one batched pass
        activations, _ = self.forward(np.asarray(inputs))
        y = self._one_hot(targets)
        return float(((activations[-1] - y) ** 2).sum(axis=1).mean())

    def accuracy(self, inputs, targets) -> float:
        """Fraction of samples whose highest output matches the target."""
        activations, _ = self.forward(np.asarray(inputs))
        return float((activations[-1].argmax(axis=1) == np.asarray(targets)).mean())

    def backprop(
        self, activations: list[np.ndarray], zs: list[np.ndarray], targets
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Returns gradients shaped like self.weights and self.biases, averaged
        # over the batch. weights[l] maps activations[l] -> activations[l + 1],
        # and zs[l] is the weighted input for activations[l + 1].
        # Every array here has one row per sample in the batch.

        # One-hot desired outputs, (batch, outputDim).
        y = self._one_hot(targets)
        batch = y.shape[0]

        step_w = [np.zeros_like(w) for w in self.weights]
        step_b = [np.zeros_like(b) for b in self.biases]

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

        return (step_w, step_b)

    def train(
        self,
        inputs,
        targets,
        epochs: int,
        learning_rate: float = 3.0,
        batch_size: int = 32,
    ) -> None:
        # In order to train, we first have to get the cost of this given training step, calculate our gradient descent vector,
        # and then update the weights with that vector by subtracting it.
        #
        # Instead of one sample at a time we use mini-batches: each step runs
        # batch_size samples through the net as one matrix, averages their
        # gradients and takes one step. Far fewer, bigger numpy calls is what
        # makes this fast.

        X = np.asarray(inputs)
        T = np.asarray(targets)
        n = X.shape[0]
        num_batches = (n + batch_size - 1) // batch_size
        log_every = max(1, num_batches // 10)

        train_start = time.perf_counter()
        print(
            f"[train] starting: {epochs} epochs x {n} samples "
            f"(batch {batch_size}, lr {learning_rate})",
            flush=True,
        )

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            # shuffle each epoch so batches differ
            order = np.random.permutation(n)

            for b in range(num_batches):
                if b > 0 and b % log_every == 0:
                    elapsed = time.perf_counter() - epoch_start
                    done = b * batch_size
                    print(
                        f"[train] epoch {epoch + 1}/{epochs}: {done}/{n} samples "
                        f"({elapsed:.1f}s, {done / elapsed:.0f} samples/s)",
                        flush=True,
                    )

                idx = order[b * batch_size : (b + 1) * batch_size]

                # invoke the nn
                activations, zs = self.forward(X[idx])

                # do backprop
                step_w, step_b = self.backprop(activations, zs, T[idx])

                # update weights
                for l in range(len(self.weights)):
                    self.weights[l] -= learning_rate * step_w[l]
                    self.biases[l] -= learning_rate * step_b[l]

            epoch_time = time.perf_counter() - epoch_start
            total_time = time.perf_counter() - train_start
            eta = total_time / (epoch + 1) * (epochs - epoch - 1)
            print(
                f"[train] epoch {epoch + 1}/{epochs} done in {epoch_time:.1f}s "
                f"(total {total_time:.1f}s, ETA {eta:.1f}s)",
                flush=True,
            )

        print(
            f"[train] finished in {time.perf_counter() - train_start:.1f}s",
            flush=True,
        )

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
        inputDim, outputDim, hiddenDim, numHidden = (int(v) for v in data["dims"])
        net = cls(inputDim, outputDim, hiddenDim, numHidden)
        net.weights = [data[f"w{l}"] for l in range(len(net.weights))]
        net.biases = [data[f"b{l}"] for l in range(len(net.biases))]
        print(f"[load] loaded weights from {path}", flush=True)
        return net
