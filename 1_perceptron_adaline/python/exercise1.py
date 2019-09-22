from neuralNetwork.Perceptron import Perceptron

x = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

y_and = [0, 0, 0, 1]
y_or = [0, 1, 1, 1]

perceptron = Perceptron([x, x], [y_and, y_or], labels=['AND', 'OR'],
                        uses_batch=False, error_threshold=0)

perceptron.predict([1, 1])
