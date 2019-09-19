from neuralNetwork.Adaline import Adaline
from common.adaline_exercises_functions import pre_processing, generate_random_cases

X, y = pre_processing()
X_val, y_val = generate_random_cases()

perceptron = Adaline([X], [y], eta=0.001, epochs=100000,
                     uses_batch=False, error_threshold=0.00000000001,
                     validatin_X=X_val, validation_y=y_val)


X_test, y_test = generate_random_cases()

for x_row, y in zip(X_test, y_test):
    perceptron.predict(x_row)
    print("Real y: ", y, "\n")
