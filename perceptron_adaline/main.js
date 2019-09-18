
import Perceptron from './Perceptron';
import Adaline from './Adaline';

let x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

let y = [
    [0, 0],
    [0, 1],
    [0, 1],
    [1, 1],
];

perceptron = new Perceptron()
perceptron.fit(x, y)

for(let i = 0; i < x.length; i++) {
    console.log(perceptron.predict(x[i]))
}