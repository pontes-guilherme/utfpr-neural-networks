// interface IActivationFunction {
//     apply(value: number): number;
//     derivate(value: number): number;
// }

// class Sigmoid implements IActivationFunction {
//     public apply(value: number): number {
//         return (1 / (1 + value));
//     }

//     public derivate(value: number): number {
//         return value * (1 - value);
//     }
// }

// class MLP {
//     private layers: Array<Layer>;
//     private eta: number;
//     private epochs: number;

//     constructor(
//         layers: Array<Layer>,
//         eta: number,
//         epochs: number
//     ) {
//         this.layers = layers;
//         this.eta = eta;
//         this.epochs = epochs;
//     }

//     private getLastLayer(): Layer {
//         return this.layers[this.layers.length - 1];
//     }

//     public propagate(x_row: Array<number>): Array<number> {
//         for (let i in this.layers) {
//             x_row = this.layers[i].getLayerActivation(x_row);
//         }

//         return x_row;
//     }

//     public backpropagate(x_row: Array<number>, y: Array<number>) {
//         let y_pred: Array<number> = this.propagate(x_row);
//         const lastLayer: Layer = this.getLastLayer();

//         //iterar por cada neurônio da última camada
//         //e por cada saída predita simultaneamente
//         for (let n = 0; n < lastLayer.neurons.length; n++) {
//             let neuron: Neuron = lastLayer.neurons[n];
//             let y_predicted: number = y_pred[n];
//             let y_desired: number = y[n];

//             //calcula erro do neurônio
//             neuron.calculateError(x_row, y_desired);

//             //calcula delta do neurônio
//             neuron.calculateDelta(y_predicted);

//             //calcula bias do neurônio
//             neuron.calculateBias(neuron.getError());

//             // lastLayer.calculateNeuronsProps(x_row, y, y_pred);
//         }


//         for (let l = this.layers.length - 2; l >= 0; l++) {
//             let nextLayer: Layer = this.layers[l + 1];
//             let currentLayer: Layer = this.layers[l];

//             //TODO set current layer error  
//             //currentLayer.calculateNeuronsErrors();
            

//             //TODO set current layer delta
//             //delta = error * derivada(ativacao(w))
//         }

//         //TODO update all neuron weights from all layers

//     }

//     public train(X: Array<Array<number>>, y: Array<Array<number>>): void {
//         for (let epoch = 0; epoch < this.epochs; epoch++) {
//             //TODO treinamento 
//             //forward

//             //backprop


//         }
//     }

//     public predict(x_row: Array<number>): Array<number> {
//         return this.propagate(x_row);
//     }

// }

// class Layer {
//     private n_inputs: number;
//     private n_neurons: number;
//     private eta: number;
//     private activationFunction: IActivationFunction;

//     public neurons: Array<Neuron>;

//     constructor(
//         n_inputs: number,
//         n_neurons: number,
//         activationFunction: IActivationFunction,
//     ) {
//         this.n_inputs = n_inputs;
//         this.n_neurons = n_neurons;
//         this.activationFunction = activationFunction;

//         this.neurons = this.initNeurons();
//     }

//     private initNeurons(): Array<Neuron> {
//         let neurons: Array<Neuron> = [];

//         for (let i = 0; i < this.n_neurons; i++) {
//             neurons.push(new Neuron(this.n_inputs,
//                 this.eta, this.activationFunction));
//         }

//         return neurons;
//     }

//     public getLayerActivation(x_row: Array<number>): Array<number> {
//         let predictions: Array<number> = [];

//         for (let i = 0; i < this.n_neurons; i++) {
//             predictions.push(this.neurons[i].predict(x_row));
//         }

//         return predictions;
//     }

//     public calculateNeuronsProps(
//         x_row: Array<number>,
//         y_d: Array<number>,
//         y_pred: Array<number>
//     ) {
//         for (let i in this.neurons) {
//             let neuron: Neuron = this.neurons[i];

//             neuron.setError(neuron.calculateError(x_row, y_d));
//             neuron.setDelta(neuron.calculateDelta(y_pred));
//             neuron.setBias(neuron.calculateBias(neuron.getError()));
//         }
//     }

//     //TODO
//     public calculateNeuronsErrors(x_row: Array<number>, y_d: Array<number>): void {
//         for (let i in this.neurons) {
//             let neuron: Neuron = this.neurons[i];

//             //neuron.setError(neuron.calculateError(x_row, y_d));
//         }
//     }

//     public getSumNeuronsErrors(): number {
//         let totalError: number = 0;

//         for (let i in this.neurons) {
//             let neuron: Neuron = this.neurons[i];
//             totalError += neuron.getError();
//         }

//         return totalError;
//     }

//     public getSumNeuronsDeltas(): number {
//         let totalDelta: number = 0;

//         for (let i in this.neurons) {
//             let neuron: Neuron = this.neurons[i];
//             totalDelta += neuron.getDelta();
//         }

//         return totalDelta;
//     }

// }

// class Neuron {

//     private eta: number;
//     private n_inputs: number;
//     private activationFunction: IActivationFunction;

//     private weights: number[];
//     private bias: number;
//     private error: number;
//     private delta: number;

//     constructor(
//         n_inputs: number,
//         eta: number,
//         activationFunction: IActivationFunction
//     ) {

//         this.bias = 1;
//         this.delta = 0;
//         this.error = 0;

//         this.n_inputs = n_inputs;
//         this.eta = eta;
//         this.activationFunction = activationFunction;

//         this.weights = this.initWeights(n_inputs);
//     }

//     private initWeights(size: number) {
//         let arr: Array<number> = [];

//         for (let i = 0; i < size; i++) {
//             arr.push(Math.random());
//         }

//         return arr;
//     }

//     public getBias(): number {
//         return this.bias;
//     }

//     public setBias(bias: number): void {
//         this.bias = bias;
//     }

//     public getError(): number {
//         return this.error;
//     }

//     public setError(error: number): void {
//         this.error = error;
//     }

//     public getDelta(): number {
//         return this.delta;
//     }

//     public setDelta(delta: number): void {
//         this.delta = delta;
//     }

//     public getWeights(): Array<number> {
//         return this.weights;
//     }


//     public calculateBias(error: number): number {
//         return this.bias + error * this.eta;
//     }

//     public calculateError(x_row: Array<number>, y_d: number): number {
//         return y_d - this.predict(x_row);
//     }

//     public calculateDelta(y_pred: number): number {
//         return this.error * this.activationFunction.derivate(y_pred);
//     }

//     public updateWeights(x_row: Array<number>, error: number): void {
//         for (let i = 0; i < this.n_inputs; i++) {
//             this.weights[i] = this.weights[i] + (error * x_row[i] * this.eta);
//         }
//     }

//     public predict(x_row: Array<number>): number {
//         let prediction: number = 0;

//         for (let i in x_row) {
//             prediction += ((x_row[i] * this.weights[i] * this.eta) + this.bias);
//         }

//         return this.activationFunction.apply(prediction);
//     }


// }

// const layer1 = new Layer(2, 3, new Sigmoid);
// const layer2 = new Layer(3, 3, new Sigmoid);
// const layer3 = new Layer(3, 1, new Sigmoid);

// const mlp = new MLP([layer1, layer2, layer3], 0.01, 10000);

// const X = [
//     [0, 0],
//     [0, 1],
//     [1, 0],
//     [1, 1],
// ];

// const y = [[0, 0, 0, 1]];


// mlp.train(X, y);
// console.log(mlp.predict([1, 1]));