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

        self.weights = []
        self.activations = []
        self.biases = []

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

    def relu(self, input: list[float]) -> list[float]:
        return [max(0, val) for val in input]

    def step(self, curLayer: int) -> None:
        """
        will update curLayer+1 activations
        """
        curW, curA, curB = (
            self.weights[curLayer],
            self.activations[curLayer],
            self.biases[curLayer],
        )

        self.activations[curLayer + 1] = self.relu(curW @ curA + curB)

    def invoke(self, input: list) -> list:
        # seed input activations
        self.activations[0] = input
        for i in range(self.numLayers - 1):
            self.step(i)
        return self.activations[-1]

    def _cost(self, input: list, target: int):
        # Will calculate the cost of an input activation neurons to its target.
        output = self.invoke(input)

        cost_val = 0

        for i in range(self.outputDim):
            if i == target:
                cost_val += (output[i] - 1) ** 2
            else:
                cost_val += output[i] ** 2

        return cost_val

    def cost(self, inputs: list[list], targets: list[int]):
        avg_cost = 0

        for i in range(len(inputs)):
            avg_cost += self._cost(inputs[i], targets[i])

        return avg_cost / len(inputs)
