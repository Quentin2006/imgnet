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

        self.numLayers = 2 + numHidden
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.numHidden = numHidden
        self.hiddenDim = hiddenDim

        self.weights: list[np.ndarray] = []
        self.activations: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

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

    def sig(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def sig_prime(self, z: np.ndarray) -> np.ndarray:
        s = self.sig(z)
        return s * (1 - s)

    def step(self, curLayer: int) -> np.ndarray:
        """
        will update curLayer+1 activations
        """
        curW, curA, curB = (
            self.weights[curLayer],
            self.activations[curLayer],
            self.biases[curLayer],
        )

        z = curW @ curA + curB

        self.activations[curLayer + 1] = self.sig(z)

        return z

    def forward(self, input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # seed input activations
        self.activations[0] = input

        zs = []

        for i in range(self.numLayers - 1):
            zs.append(self.step(i))

        return (self.activations, zs)

    def __cost__(self, input: np.ndarray, target: int) -> float:
        # Will calculate the cost of an input activation neurons to its target.
        activations, _ = self.forward(input)
        perdiction = activations[-1]

        cost_val = 0

        for i in range(self.outputDim):
            if i == target:
                cost_val += (perdiction[i] - 1) ** 2
            else:
                cost_val += perdiction[i] ** 2

        return cost_val

    def __cost_given_output__(self, outputs: np.ndarray, target: int) -> float:
        perdiction = outputs

        cost_val = 0

        for i in range(self.outputDim):
            if i == target:
                cost_val += (perdiction[i] - 1) ** 2
            else:
                cost_val += perdiction[i] ** 2

        return cost_val

    def cost(self, inputs: list[np.ndarray], targets: list[int]) -> float:
        avg_cost = 0

        for i in range(len(inputs)):
            avg_cost += self.__cost__(inputs[i], targets[i])

        return avg_cost / len(inputs)

    def backprop(
        self, activations: list[np.ndarray], zs: list[np.ndarray], target: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Returns gradients shaped like self.weights and self.biases.
        # weights[l] maps activations[l] -> activations[l + 1], and zs[l] is
        # the weighted input for activations[l + 1].

        # One-hot desired output.
        y = np.zeros(self.outputDim)
        y[target] = 1.0

        step_w = [np.zeros_like(w) for w in self.weights]
        step_b = [np.zeros_like(b) for b in self.biases]

        # dC/da for the output layer.
        dC_da = 2 * (np.asarray(activations[-1]) - y)

        # Walk backwards through the layers.
        for l in reversed(range(len(self.weights))):
            # dC/dz_j = sig'(z_j) * dC/da_j
            delta = self.sig_prime(zs[l]) * dC_da

            # dC/dw_jk = a_k * dC/dz_j
            step_w[l] = np.outer(delta, activations[l])
            # dC/db_j = dC/dz_j
            step_b[l] = delta

            # dC/da_k for the previous layer = sum_j w_jk * dC/dz_j
            dC_da = self.weights[l].T @ delta

        return (step_w, step_b)

    def train(
        self,
        inputs: list[np.ndarray],
        targets: list[int],
        epochs: int,
        learning_rate: float = 0.1,
    ) -> None:
        # In order to train, we first have to get the cost of this given training step, calculate our gradient descent vector,
        # and then update the weights with that vector by subtracting it.

        for i in range(epochs):
            for i in range(len(inputs)):
                input, target = inputs[i], targets[i]

                # invoke the nn
                activations, zs = self.forward(input)

                # do backprop
                step_w, step_b = self.backprop(activations, zs, target)

                # update weights
                for l in range(len(self.weights)):
                    self.weights[l] -= learning_rate * step_w[l]
                    self.biases[l] -= learning_rate * step_b[l]
