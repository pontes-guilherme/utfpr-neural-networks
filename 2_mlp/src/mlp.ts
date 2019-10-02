interface IActivationFunction {
    apply(value: number): number;
    derivate(value: number): number;
}

class Sigmoid implements IActivationFunction {
    public apply(value: number): number {
        return (1 / (1 + value));
    }

    public derivate(value: number): number {
        return value * (1 - value);
    }
}

class MLP {
    private layers: Array<Layer>;
    private eta: number;
    private epochs: number;

    constructor(
        layers: Array<Layer>,
        eta: number,
        epochs: number
    ) {
        this.layers = layers;
        this.eta = eta;
        this.epochs = epochs;
    }

    private getLastLayer(): Layer {
        return this.layers[this.layers.length - 1];
    }

    public propagate(x_row: Array<number>): Array<number> {
        for (let i in this.layers) {
            x_row = this.layers[i].getLayerActivation(x_row);
        }

        return x_row;
    }

    public backpropagate(x_row: Array<number>, y: Array<number>) {
        let y_pred: Array<number> = this.propagate(x_row);
        const lastLayer: Layer = this.getLastLayer();

        //TODO

    }

    public train(X: Array<Array<number>>, y: Array<number>): void {

    }

    public predict(x_row: Array<number>): Array<number> {
        return this.getLastLayer().getLayerActivation(x_row);
    }

}

class Layer {
    private n_inputs: number;
    private n_neurons: number;
    private eta: number;
    private activationFunction: IActivationFunction;

    public neurons: Array<Neuron>;

    constructor(
        n_inputs: number,
        n_neurons: number,
        eta: number,
        activationFunction: IActivationFunction,
    ) {
        this.n_inputs = n_inputs;
        this.n_neurons = n_neurons;
        this.eta = eta;
        this.activationFunction = activationFunction;

        this.neurons = this.initNeurons();
    }

    private initNeurons(): Array<Neuron> {
        let neurons: Array<Neuron> = [];

        for (let i = 0; i < this.n_neurons; i++) {
            neurons.push(new Neuron(this.n_inputs,
                this.eta, this.activationFunction));
        }

        return neurons;
    }

    public getLayerActivation(x_row: Array<number>): Array<number> {
        let predictions: Array<number> = [];

        for (let i = 0; i < this.n_neurons; i++) {
            predictions.push(this.neurons[i].predict(x_row));
        }

        return predictions;
    }

}

class Neuron {
    private bias: number;
    private delta: number;
    private eta: number;
    private n_inputs: number;
    private activationFunction: IActivationFunction;

    private weights: number[];

    constructor(
        n_inputs: number,
        eta: number,
        activationFunction: IActivationFunction
    ) {

        this.bias = 1;
        this.delta = 0;

        this.n_inputs = n_inputs;
        this.eta = eta;
        this.activationFunction = activationFunction;

        this.weights = this.initWeights(n_inputs);
    }

    private initWeights(size: number) {
        let arr: Array<number> = [];

        for (let i = 0; i < size; i++) {
            arr.push(Math.random());
        }

        return arr;
    }

    public setBias(error: number): void {
        this.bias = this.bias * error * this.eta;
    }

    public updateWeights(x_row: Array<number>, error: number): void {
        for (let i = 0; i < this.n_inputs; i++) {
            this.weights[i] = this.weights[i] + (error * x_row[i] * this.eta);
        }
    }

    public predict(x_row: Array<number>): number {
        let prediction: number = 0;

        for (let i in x_row) {
            prediction += x_row[i] * this.weights[i] * this.eta;
        }

        //return prediction;

        return this.activationFunction.apply(prediction);
    }


}

let layer = new Layer(2, 3, 0.01, new Sigmoid);