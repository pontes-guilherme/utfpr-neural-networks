from neuralNetwork.Adaline import Adaline
from common.adaline_exercises_functions import pre_processing

X, y = pre_processing()

perceptron = Adaline([X], [y],
                     uses_batch=False, error_threshold=0.0000000001)

for x_row in X:
    perceptron.predict(x_row)
