from neuralNetwork.Adaline import Adaline
from common.adaline_exercises_functions import pre_processing

X, y = pre_processing()

perceptron = Adaline([X], [y], eta=0.001, epochs=100000,
                     uses_batch=True, error_threshold=0.00000000001)

for x_row in X:
    perceptron.predict(x_row)
