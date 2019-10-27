interface IActivationFunction {
    apply(value: number): number;
    derivate(value: number): number;
}

class Sigmoid implements IActivationFunction {
    public apply(value: number): number {
        const result: number = (1 / (1 + Math.pow(Math.E, -value)));

        return result;
    }

    public derivate(value: number): number {
        const result: number = value * (1 - value);

        return result;
    }
}

class MLP {
    private layers: Layer[];
    private eta: number;
    private epochs: number;

    constructor(
        layers: Layer[],
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

    public propagate(x_row: number[]): number[] {
        let _x_row: number[] = x_row;

        for (let i in this.layers) {
            _x_row = this.layers[i].calculateLayerActivation(_x_row);
            this.layers[i].setActivations(_x_row);
        }

        return _x_row;
    }

    public train(X: number[][], y: number[][]): void {
        const X_NORM: number[][] = X;
        const p_size: number = X_NORM.length;

        for (let epoch = 0; epoch < this.epochs; epoch++) {
            let sum_of_errors: number = 0;

            //limpa os delta weights no início de cada época
            for (let l = this.layers.length - 1; l >= 0; l--) {
                let currentLayer: Layer = this.layers[l];
                currentLayer.clearDeltaWeights();
            }

            const lastLayer: Layer = this.getLastLayer();
            const penultimateLayer: Layer = this.layers[this.layers.length - 2];

            //para cada padrão de treinamento
            for (let p = 0; p < X_NORM.length; p++) {
                const x_row: number[] = X_NORM[p];
                const y_pred: number[] = this.propagate(x_row);

                //iterar por cada neurônio da última camada
                for (let n = 0; n < lastLayer.getNNeurons(); n++) {
                    const y_desired_neuron: number[] = y[n];
                    const y_desired: number = y_desired_neuron[p];
                    const y_predicted: number = y_pred[n];

                    //calcula erro do neurônio
                    const error: number = y_desired - y_predicted;
                    sum_of_errors += (error ** 2);
                    lastLayer.setNeuronError(n, error);

                    //calcula delta do neurônio
                    const delta: number = lastLayer.calculateNeuronDelta(n, y_predicted);
                    lastLayer.setNeuronDelta(n, delta);

                    //calcula bias do neurônio
                    const bias: number = lastLayer.calculateNeuronBias(n, error);
                    lastLayer.setNeuronBias(n, bias);

                    const neuron_delta_weights: number[] = lastLayer.getNeuronDeltaWeights(n);
                    //setar delta_weight baseado na saída da última camada
                    let delta_weight = [];
                    const penultimate_layer_x_row: number[] = penultimateLayer.getActivations();

                    for (let j in penultimate_layer_x_row) {
                        let x_from_previous_layer: number = penultimate_layer_x_row[j];

                        const previous_delta = neuron_delta_weights.length ? neuron_delta_weights[j] : 0;

                        const newDeltaWeight = (previous_delta + (this.eta * delta * x_from_previous_layer));
                        delta_weight.push(newDeltaWeight);
                    }

                    lastLayer.setNeuronDeltaWeights(n, delta_weight);
                }


                for (let l = this.layers.length - 2; l >= 0; l--) {
                    let nextLayer: Layer = this.layers[l + 1];
                    let currentLayer: Layer = this.layers[l];

                    let x_row_considered: number[];

                    if (l > 0) {
                        let previousLayer: Layer = this.layers[l - 1];
                        x_row_considered = previousLayer.getActivations();
                    } else {
                        x_row_considered = x_row;
                    }

                    //iterar por cada neurônio da camada
                    for (let n = 0; n < currentLayer.getNNeurons(); n++) {
                        let error: number = 0;

                        for (let n_next_idx = 0; n_next_idx < nextLayer.getNNeurons(); n_next_idx++) {
                            let delta = nextLayer.getNeuronDelta(n_next_idx);

                            for (let w_next_idx = 0; w_next_idx < nextLayer.getNeuronWeights(n_next_idx).length; w_next_idx++) {
                                const w_next = nextLayer.getNeuronWeights(n_next_idx)[w_next_idx];
                                error += (delta * w_next);
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

                        const neuron_delta_weights: number[] = currentLayer.getNeuronDeltaWeights(n);
                        //setar delta_weight baseado na saída da última camada
                        let delta_weight = [];
                        for (let j in x_row_considered) {
                            let x_from_previous_layer: number = x_row_considered[j];

                            const previous_delta = neuron_delta_weights.length ? neuron_delta_weights[j] : 0;

                            const newDeltaWeight = (previous_delta + (this.eta * currentLayer.getNeuronError(n) * x_from_previous_layer));
                            delta_weight.push(newDeltaWeight);

                        }

                        currentLayer.setNeuronDeltaWeights(n, delta_weight);
                    }
                }

            }

            //update all neuron weights from all layers
            for (let l = 0; l < this.layers.length; l++) {
                let currentLayer: Layer = this.layers[l];

                for (let n = 0; n < currentLayer.getNNeurons(); n++) {
                    let neuronWeights: number[] = currentLayer.getNeuronWeights(n);
                    const neuronDeltaWeights: number[] = currentLayer.getNeuronDeltaWeights(n);

                    for (let i in neuronDeltaWeights) {
                        neuronWeights[i] += (neuronDeltaWeights[i] / p_size);
                    }

                    currentLayer.setNeuronWeights(n, neuronWeights);
                }
            }

            console.log(`epoch ${epoch}: sum error = ${sum_of_errors / p_size}`);

        }
    }

    public predict(x_row: number[]): number[] {
        return this.propagate(x_row);
    }

}

class Layer {
    private n_inputs: number;
    private n_neurons: number;
    private eta: number;
    public activationFunction: IActivationFunction;

    private weights: number[][] = new Array();
    private biases: number[] = new Array();
    private errors: number[] = new Array();
    private deltas: number[] = new Array();
    private activations: number[] = new Array();
    private delta_weights: number[][] = new Array();

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
        this.delta_weights = new Array();
        this.errors = new Array();
        this.biases = new Array();
        this.deltas = new Array();
        this.activations = new Array();

        for (let neuron = 0; neuron < this.n_neurons; neuron++) {
            this.weights[neuron] = new Array();
            this.delta_weights[neuron] = new Array();

            for (let input = 0; input < this.n_inputs; input++) {
                this.weights[neuron].push(Math.random() * 2 - 1);
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

    public getNeuronWeights(neuron_idx: number): number[] {
        return this.weights[neuron_idx];
    }

    public getNeuronError(neuron_idx: number): number {
        return this.errors[neuron_idx];
    }

    public getNeuronDelta(neuron_idx: number): number {
        return this.deltas[neuron_idx];
    }

    public getNeuronDeltaWeights(neuron_idx: number): number[] {
        return this.delta_weights[neuron_idx];
    }

    public getNeuronBias(neuron_idx: number): number {
        return this.biases[neuron_idx];
    }

    public getActivations(): number[] {
        return this.activations;
    }


    public setEta(eta: number): void {
        this.eta = eta;
    }

    public setActivations(activations: number[]): void {
        this.activations = activations;
    }

    public setNeuronError(neuron_idx: number, value: number): void {
        this.errors[neuron_idx] = value;
    }

    public setNeuronDelta(neuron_idx: number, value: number): void {
        this.deltas[neuron_idx] = value;
    }

    public setNeuronBias(neuron_idx: number, value: number): void {
        this.biases[neuron_idx] = value;
    }

    public setNeuronWeights(neuron_idx: number, weights: number[]): void {
        this.weights[neuron_idx] = weights;
    }

    public setNeuronDeltaWeights(neuron_idx: number, deltas: number[]): void {
        this.delta_weights[neuron_idx] = deltas;
    }

    public clearDeltaWeights() {
        this.delta_weights = new Array();

        for (let neuron = 0; neuron < this.n_neurons; neuron++) {
            this.delta_weights[neuron] = new Array();
        }
    }

    public calculateLayerActivation(x_row: number[]): number[] {
        let activations: number[] = new Array();

        for (let neuron = 0; neuron < this.n_neurons; neuron++) {
            let activation = this.calculateNeuronActivation(neuron, x_row);
            activations.push(activation);
        }

        return activations;
    }


    public calculateNeuronActivation(neuron_idx: number, x_row: number[]): number {
        let prediction: number = 0;

        for (let i in x_row) {
            prediction += (x_row[i] * this.weights[neuron_idx][i] * this.eta);
        }

        prediction += this.biases[neuron_idx];

        let activation = this.activationFunction.apply(prediction);

        return activation;
    }

    public calculateNeuronBias(neuron_idx: number, error: number): number {
        const bias = this.biases[neuron_idx] + (error * this.eta);

        return bias;
    }

    public calculateNeuronDelta(neuron_idx: number, y_pred: number): number {
        const delta: number = this.errors[neuron_idx] * this.activationFunction.derivate(y_pred);

        return delta;
    }

}


const layer1 = new Layer(2, 3, new Sigmoid);
const layer2 = new Layer(3, 1, new Sigmoid);

const mlp = new MLP([layer1, layer2], 0.01, 100000);

const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

const y = [[0, 1, 1, 0]];

mlp.train(X, y);
console.log(mlp.predict([0, 0]));
console.log(mlp.predict([0, 1]));
console.log(mlp.predict([1, 0]));
console.log(mlp.predict([1, 1]));