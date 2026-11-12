import time
from pathlib import Path

import numpy as np

# CuPy is a drop-in replacement for numpy that runs on the GPU.
# If it isn't installed (or there's no GPU) we fall back to numpy.
try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

# float32 is plenty for this net, and consumer GPUs are ~64x slower at float64
DTYPE = np.float32


class NeuralNet:
    def __init__(
        self,
        inputDim: int,
        outputDim: int,
        hiddenDim: int,
        numHidden: int,
        device: str = "auto",
    ) -> None:
        """
        we can encode nerual nets by using matricies.

        from one layer to another, we can define the activation of the next layer nerons by

        sig(weight_mat * activation_vec + bias_vec) = next_layer_activations

        device: "gpu", "cpu", or "auto" (gpu if available)
        """

        if device == "auto":
            device = "gpu" if GPU_AVAILABLE else "cpu"
        if device == "gpu" and not GPU_AVAILABLE:
            raise RuntimeError("device='gpu' requested but CuPy/CUDA is unavailable")
        self.device = device
        # xp is either numpy or cupy; every array op goes through it
        self.xp = cp if device == "gpu" else np

        rng = np.random.default_rng()

        self.numLayers = 2 + numHidden
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.numHidden = numHidden
        self.hiddenDim = hiddenDim

        self.weights: list = []
        self.activations: list = []
        self.biases: list = []

        # populate for input layer
        self.weights.append(rng.uniform(-1, 1, (hiddenDim, inputDim)))
        self.activations.append(np.zeros(inputDim))
        self.biases.append(np.zeros(hiddenDim))

        # populate hiden layers
        for i in range(numHidden):
            self.activations.append(np.zeros(hiddenDim))
            if i < numHidden - 1:
                self.weights.append(rng.uniform(-1, 1, (hiddenDim, hiddenDim)))
                self.biases.append(np.zeros(hiddenDim))

        # populate for output layer
        self.weights.append(rng.uniform(-1, 1, (outputDim, hiddenDim)))
        self.activations.append(np.zeros(outputDim))
        self.biases.append(np.zeros(outputDim))

        # move everything onto the chosen device
        self.weights = [self._to_device(w) for w in self.weights]
        self.biases = [self._to_device(b) for b in self.biases]
        self.activations = [self._to_device(a) for a in self.activations]

    # ---------- device helpers ----------

    def _to_device(self, arr) -> "np.ndarray":
        return self.xp.asarray(arr, dtype=DTYPE)

    def _to_host(self, arr) -> np.ndarray:
        return cp.asnumpy(arr) if self.xp is cp else np.asarray(arr)

    def _sync(self) -> None:
        # GPU work is async; wait for it so timings are honest
        if self.xp is cp:
            cp.cuda.Stream.null.synchronize()

    # ---------- math ----------

    def sig(self, z):
        # clip so exp() can't overflow float32 (sig is already 0/1 out there)
        return 1 / (1 + self.xp.exp(-self.xp.clip(z, -60, 60)))

    def sig_prime(self, z):
        s = self.sig(z)
        return s * (1 - s)

    def step(self, curLayer: int):
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

    def forward(self, input) -> tuple[list, list]:
        # seed input activations (a single (inputDim,) vector or a (batch, inputDim) matrix)
        self.activations[0] = self._to_device(input)

        zs = []

        for i in range(self.numLayers - 1):
            zs.append(self.step(i))

        return (self.activations, zs)

    def predict(self, input) -> np.ndarray:
        """Output layer activations as a plain numpy array (handy for display code)."""
        activations, _ = self.forward(input)
        return self._to_host(activations[-1])

    def _one_hot(self, targets):
        targets = self.xp.asarray(targets).reshape(-1)
        y = self.xp.zeros((targets.shape[0], self.outputDim), dtype=DTYPE)
        y[self.xp.arange(targets.shape[0]), targets] = 1.0
        return y

    def __cost__(self, input, target: int) -> float:
        # Will calculate the cost of an input activation neurons to its target.
        activations, _ = self.forward(input)
        return self.__cost_given_output__(activations[-1], target)

    def __cost_given_output__(self, outputs, target: int) -> float:
        y = self._one_hot([target])[0]
        return float(((self.xp.asarray(outputs) - y) ** 2).sum())

    def cost(self, inputs, targets) -> float:
        # average squared error over every sample, done as one batched pass
        activations, _ = self.forward(self._stack(inputs))
        y = self._one_hot(targets)
        return float(((activations[-1] - y) ** 2).sum(axis=1).mean())

    def accuracy(self, inputs, targets, batch_size: int = 10000) -> float:
        """Fraction of samples whose highest output matches the target."""
        X = self._stack(inputs)
        t = self.xp.asarray(targets)
        correct = 0
        for s in range(0, X.shape[0], batch_size):
            activations, _ = self.forward(X[s : s + batch_size])
            correct += int((activations[-1].argmax(axis=1) == t[s : s + batch_size]).sum())
        return correct / X.shape[0]

    def backprop(self, activations: list, zs: list, targets) -> tuple[list, list]:
        # Returns gradients shaped like self.weights and self.biases, averaged
        # over the batch. weights[l] maps activations[l] -> activations[l + 1],
        # and zs[l] is the weighted input for activations[l + 1].
        # Every array here has one row per sample in the batch.

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

        return (step_w, step_b)

    def _stack(self, inputs):
        # list of vectors -> one (n, inputDim) matrix on the device
        if isinstance(inputs, list):
            inputs = np.stack(inputs)
        return self._to_device(inputs)

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
        # gradients and takes one step. That's what makes the GPU worth it.

        X = self._stack(inputs)  # copied to the GPU once, up front
        T = self.xp.asarray(targets)
        n = X.shape[0]
        num_batches = (n + batch_size - 1) // batch_size
        log_every = max(1, num_batches // 10)

        self._sync()
        train_start = time.perf_counter()
        print(
            f"[train] starting on {self.device}: {epochs} epochs x {n} samples "
            f"(batch {batch_size}, lr {learning_rate})",
            flush=True,
        )

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            # shuffle each epoch so batches differ
            order = self.xp.random.permutation(n)

            for b in range(num_batches):
                if b > 0 and b % log_every == 0:
                    self._sync()
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

            self._sync()
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

    # ---------- saving / loading ----------

    def save(self, path: str | Path) -> None:
        """Write shape info, weights and biases to a single .npz file."""
        arrays = {
            "dims": np.array(
                [self.inputDim, self.outputDim, self.hiddenDim, self.numHidden]
            )
        }
        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{l}"] = self._to_host(w)
            arrays[f"b{l}"] = self._to_host(b)
        np.savez(path, **arrays)
        print(f"[save] wrote weights to {path}", flush=True)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "NeuralNet":
        """Rebuild a network from a file written by save()."""
        data = np.load(path)
        inputDim, outputDim, hiddenDim, numHidden = (int(v) for v in data["dims"])
        net = cls(inputDim, outputDim, hiddenDim, numHidden, device=device)
        net.weights = [net._to_device(data[f"w{l}"]) for l in range(len(net.weights))]
        net.biases = [net._to_device(data[f"b{l}"]) for l in range(len(net.biases))]
        print(f"[load] loaded weights from {path} ({net.device})", flush=True)
        return net
