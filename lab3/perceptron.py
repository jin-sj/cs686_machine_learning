import random
import matplotlib.pyplot as plt
import numpy as np

from classifier import classifier
from sklearn.datasets import make_classification

SEED = 42

class Perceptron(classifier):
    def __init__(self, learning_rate=0.001, epochs=10, seed=SEED, activation="relu"):
        super().__init__()
        random.seed(seed)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

        if activation == "relu":
            self.activation = self.relu
        elif activation == "sigmoid":
            self.activation = self.sigmoid

    def fit(self, X, Y):
        super().fit(X, Y)

        self.weights = np.zeros(X.shape[1])
        self.bias = 1
        for i in range(self.epochs):
            print("Epoch: %d, weights: %s, bias: %s" % (i, self.weights, self.bias))
            for j in range(X.shape[0]):
                self.sgd(X[j], Y[j])

    def _predict(self, x):
        super().predict(x)
        y = np.sum(self.weights * x + self.bias)
        return self.activation(y)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            predictions[i] = self._predict(X[i])
        return predictions

    def sgd(self, x, y):
        prediction = self._predict(x)
        new_bias = self.bias + self.learning_rate * (y - prediction) * self.bias
        new_weights = self.weights + self.learning_rate * (y - prediction) * x
        self.bias = new_bias
        self.weights = new_weights

    def relu(self, y):
        if y < 0:
            return 0
        return 1

    def sigmoid(self, y):
        y = np.exp(y) / (np.exp(y) + 1)
        return y

def main():
    X, Y = make_classification(200, 2, 2, 0, weights=[.5, .5], random_state=SEED)
    perceptron = Perceptron(epochs=100, activation="sigmoid")
    perceptron.fit(X, Y)
    predictions = perceptron.predict(X)
    plt.scatter(X[:,0], X[:,1], c=Y)
    x = np.linspace(-3.0, 3.0, num=100)
    y = perceptron.weights[0] * x + perceptron.bias
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()
