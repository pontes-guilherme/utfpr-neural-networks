interface Throwable {
    getMessage(): String;
}

class Exception implements Throwable {
    private message: String;

    constructor(message: String) {
        this.message = message
    }

    public getMessage(): String {
        return this.message;
    }
}

interface IActivationFunction {
    apply(value: number): number;
    derivate(value: number): number;
}

class Sigmoid implements IActivationFunction {
    public apply(value: number): number {
        const result: number = (1 / (1 + value));

        if (isNaN(result)) {
            throw new Exception('Resulted in NaN');
        }

        return result;
    }

    public derivate(value: number): number {
        const result: number = value * (1 - value);

        if (isNaN(result)) {
            throw new Exception('Resulted in NaN');
        }

        return result;
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

        for (let layer in this.layers) {
            this.layers[layer].setEta(eta);
        }
    }

    private getLastLayer(): Layer {
        return this.layers[this.layers.length - 1];
    }

    public propagate(x_row: Array<number>): Array<number> {
        let _x_row: Array<number> = x_row;

        for (let i in this.layers) {
            _x_row = this.layers[i].calculateLayerActivation(_x_row);
        }

        return _x_row;
    }

    public train(X: Array<Array<number>>, y: Array<Array<number>>): void {
        for (let epoch = 0; epoch < this.epochs; epoch++) {

            const lastLayer: Layer = this.getLastLayer();

            //iterar por cada neurônio da última camada
            for (let n = 0; n < lastLayer.getNNeurons(); n++) {

                const y_desired_neuron: Array<number> = y[n];

                //para cada padrão de treinamento
                for (let p = 0; p < X.length; p++) {
                    const x_row: Array<number> = X[p];
                    const y_desired: number = y_desired_neuron[p];

                    const y_pred: Array<number> = this.propagate(x_row);

                    const y_predicted: number = y_pred[n];


                    //calcula erro do neurônio
                    const error: number = lastLayer.calculateNeuronError(n, x_row, y_desired);
                    lastLayer.setNeuronError(n, error);

                    //calcula delta do neurônio
                    const delta: number = lastLayer.calculateNeuronDelta(n, y_predicted);
                    lastLayer.setNeuronDelta(n, delta);

                    //calcula bias do neurônio
                    const bias: number = lastLayer.calculateNeuronBias(n, error);
                    lastLayer.setNeuronBias(n, bias);
                }

            }

            for (let l = this.layers.length - 2; l >= 0; l--) {
                let nextLayer: Layer = this.layers[l + 1];
                let currentLayer: Layer = this.layers[l];

                //iterar por cada neurônio da camada
                for (let n = 0; n < currentLayer.getNNeurons(); n++) {

                    //para cada padrão de treinamento
                    for (let p = 0; p < X.length; p++) {
                        const x_row: Array<number> = X[p];

                        let error = 0;

                        for (let n_next = 0; n_next < nextLayer.getNNeurons(); n_next++) {
                            let delta = nextLayer.getNeuronDelta(n_next);

                            for (let w_next = 0; w_next < nextLayer.getNeuronWeights(n_next).length; w_next++) {
                                const w = nextLayer.getNeuronWeights(n_next)[w_next];
                                error += (delta * w);
                            }
                        }

                        currentLayer.setNeuronError(n, error);

                        //calcula delta do neurônio
                        let delta = error *
                            currentLayer.activationFunction.derivate(currentLayer.calculateNeuronActivation(n, x_row));
                        currentLayer.setNeuronDelta(n, delta);

                        //calcula bias do neurônio
                        let bias = currentLayer.calculateNeuronBias(n, error);
                        currentLayer.setNeuronBias(n, bias);
                    }

                }
            }

            //update all neuron weights from all layers
            for (let l = this.layers.length - 1; l >= 0; l--) {
                let currentLayer: Layer = this.layers[l];

                for (let n = 0; n < currentLayer.getNNeurons(); n++) {
                    let neuronWeights: Array<number> = currentLayer.getNeuronWeights(n);

                    for (let p = 0; p < X.length; p++) {
                        const x_row: Array<number> = X[p];

                        for (let i = 0; i < x_row.length; i++) {
                            const x_i: number = x_row[i];
                            const currentWeight: number = neuronWeights[i];

                            neuronWeights[i] = currentWeight +
                                (this.eta * currentLayer.getNeuronError(n) * currentWeight * x_i);
                        }
                    }

                    currentLayer.setNeuronWeights(n, neuronWeights);
                }
            }

        }
    }

    public predict(x_row: Array<number>): Array<number> {
        return this.propagate(x_row);
    }

}

class Layer {
    private n_inputs: number;
    private n_neurons: number;
    private eta: number;
    public activationFunction: IActivationFunction;

    private weights: Array<Array<number>> = new Array();
    private biases: Array<number> = new Array();
    private errors: Array<number> = new Array();
    private deltas: Array<number> = new Array();

    constructor(
        n_inputs: number,
        n_neurons: number,
        activationFunction: IActivationFunction,
    ) {
        this.n_inputs = n_inputs;
        this.n_neurons = n_neurons;
        this.activationFunction = activationFunction;
        this.eta = 0.01;

        this.initNeurons();
    }

    private initNeurons(): void {
        this.weights = new Array();
        this.errors = new Array();
        this.biases = new Array();
        this.deltas = new Array();

        for (let neuron = 0; neuron < this.n_neurons; neuron++) {
            this.weights[neuron] = new Array();

            for (let input = 0; input < this.n_inputs; input++) {
                this.weights[neuron].push(Math.random());
            }

            this.biases[neuron] = 1;
            this.errors[neuron] = 0;
            this.deltas[neuron] = 0;
        }
    }


    public getNInputs(): number {
        return this.n_inputs
    }

    public getNNeurons(): number {
        return this.n_neurons
    }

    public getNeuronWeights(neuron_idx: number): Array<number> {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        return this.weights[neuron_idx];
    }

    public getNeuronError(neuron_idx: number): number {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        return this.errors[neuron_idx];
    }

    public getNeuronDelta(neuron_idx: number): number {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        return this.deltas[neuron_idx];
    }

    public getNeuronBias(neuron_idx: number): number {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        return this.biases[neuron_idx];
    }


    public setEta(eta: number): void {
        this.eta = eta;
    }

    public setNeuronError(neuron_idx: number, value: number): void {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        this.errors[neuron_idx] = value;
    }

    public setNeuronDelta(neuron_idx: number, value: number): void {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        this.deltas[neuron_idx] = value;
    }

    public setNeuronBias(neuron_idx: number, value: number): void {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        this.biases[neuron_idx] = value;
    }

    public setNeuronWeights(neuron_idx: number, weights: Array<number>): void {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        this.weights[neuron_idx] = weights;
    }

    public calculateLayerActivation(x_row: Array<number>): Array<number> {
        let activations: Array<number> = new Array();

        for (let neuron = 0; neuron < this.n_neurons; neuron++) {
            let activation = this.calculateNeuronActivation(neuron, x_row);
            activations.push(activation);
        }

        return activations;
    }


    public calculateNeuronActivation(neuron_idx: number, x_row: Array<number>): number {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        let prediction: number = 0;

        for (let i in x_row) {
            prediction += ((x_row[i] * this.weights[neuron_idx][i] * this.eta)
                + this.biases[neuron_idx]);
        }

        let activation = this.activationFunction.apply(prediction);

        if (isNaN(activation)) {
            throw new Exception('Resulted in NaN');
        }

        return activation;
    }

    public calculateNeuronBias(neuron_idx: number, error: number): number {
        const bias = this.biases[neuron_idx] + error * this.eta;

        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        if (isNaN(bias)) {
            console.log(this.n_neurons);
            console.log(neuron_idx, this.biases[neuron_idx], error, this.eta);
            throw new Exception('Resulted in NaN');
        }

        return bias;
    }

    public calculateNeuronError(neuron_idx: number, x_row: Array<number>, y_d: number): number {
        const error: number = y_d - this.calculateNeuronActivation(neuron_idx, x_row);

        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        if (isNaN(error)) {
            throw new Exception('Resulted in NaN');
        }

        return error;
    }

    public calculateNeuronDelta(neuron_idx: number, y_pred: number): number {
        const delta: number = this.errors[neuron_idx] * this.activationFunction.derivate(y_pred);

        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        if (isNaN(delta)) {
            throw new Exception('Resulted in NaN');
        }

        return delta;
    }

    public updateNeuronWeights(neuron_idx: number, x_row: Array<number>, error: number): void {

        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }

        for (let i = 0; i < this.n_inputs; i++) {
            this.weights[neuron_idx][i] = this.weights[neuron_idx][i]
                + (error * x_row[i] * this.eta);
        }
    }

}


const layer1 = new Layer(2, 3, new Sigmoid);
const layer2 = new Layer(3, 3, new Sigmoid);
const layer3 = new Layer(3, 2, new Sigmoid);

const mlp = new MLP([layer1, layer2, layer3], 0.01, 10);

const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

const y = [[0, 0, 0, 1], [0, 1, 1, 1]];


mlp.train(X, y);
console.log(mlp.predict([0, 0]));
console.log(mlp.predict([0, 1]));
console.log(mlp.predict([1, 0]));
console.log(mlp.predict([1, 1]));