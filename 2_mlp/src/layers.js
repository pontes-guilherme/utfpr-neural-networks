"use strict";
var Exception = /** @class */ (function () {
    function Exception(message) {
        this.message = message;
    }
    Exception.prototype.getMessage = function () {
        return this.message;
    };
    return Exception;
}());
var Sigmoid = /** @class */ (function () {
    function Sigmoid() {
    }
    Sigmoid.prototype.apply = function (value) {
        var result = (1 / (1 + value));
        if (isNaN(result)) {
            throw new Exception('Resulted in NaN');
        }
        return result;
    };
    Sigmoid.prototype.derivate = function (value) {
        var result = value * (1 - value);
        if (isNaN(result)) {
            throw new Exception('Resulted in NaN');
        }
        return result;
    };
    return Sigmoid;
}());
var MLP = /** @class */ (function () {
    function MLP(layers, eta, epochs) {
        this.layers = layers;
        this.eta = eta;
        this.epochs = epochs;
        for (var layer in this.layers) {
            this.layers[layer].setEta(eta);
        }
    }
    MLP.prototype.getLastLayer = function () {
        return this.layers[this.layers.length - 1];
    };
    MLP.prototype.propagate = function (x_row) {
        var _x_row = x_row;
        for (var i in this.layers) {
            _x_row = this.layers[i].calculateLayerActivation(_x_row);
            this.layers[i].setActivations(_x_row);
        }
        return _x_row;
    };
    MLP.prototype.train = function (X, y) {
        for (var epoch = 0; epoch < this.epochs; epoch++) {
            var lastLayer = this.getLastLayer();
            //iterar por cada neurônio da última camada
            for (var n = 0; n < lastLayer.getNNeurons(); n++) {
                var y_desired_neuron = y[n];
                //para cada padrão de treinamento
                for (var p = 0; p < X.length; p++) {
                    var x_row = X[p];
                    var y_desired = y_desired_neuron[p];
                    var y_pred = this.propagate(x_row);
                    var y_predicted = y_pred[n];
                    //calcula erro do neurônio
                    var error = y_desired - y_predicted;
                    lastLayer.setNeuronError(n, error);
                    //calcula delta do neurônio
                    var delta = lastLayer.calculateNeuronDelta(n, y_predicted);
                    lastLayer.setNeuronDelta(n, delta);
                    //calcula bias do neurônio
                    var bias = lastLayer.calculateNeuronBias(n, error);
                    lastLayer.setNeuronBias(n, bias);
                }
                //TODO setar delta_weight baseado na saída da última camadas
                //const delta_weight = (this.eta * currentLayer.getNeuronError(n) * x_i);
            }
            for (var l = this.layers.length - 2; l >= 0; l--) {
                var nextLayer = this.layers[l + 1];
                var currentLayer = this.layers[l];
                //iterar por cada neurônio da camada
                for (var n = 0; n < currentLayer.getNNeurons(); n++) {
                    //para cada padrão de treinamento
                    for (var p = 0; p < X.length; p++) {
                        var x_row = X[p];
                        var error = 0;
                        for (var n_next_idx = 0; n_next_idx < nextLayer.getNNeurons(); n_next_idx++) {
                            var delta_1 = nextLayer.getNeuronDelta(n_next_idx);
                            for (var w_next_idx = 0; w_next_idx < nextLayer.getNeuronWeights(n_next_idx).length; w_next_idx++) {
                                var w_next = nextLayer.getNeuronWeights(n_next_idx)[w_next_idx];
                                error += (delta_1 * w_next);
                            }
                        }
                        currentLayer.setNeuronError(n, error);
                        //calcula delta do neurônio
                        var delta = error *
                            currentLayer.activationFunction.derivate(currentLayer.calculateNeuronActivation(n, x_row));
                        currentLayer.setNeuronDelta(n, delta);
                        //calcula bias do neurônio
                        var bias = currentLayer.calculateNeuronBias(n, error);
                        currentLayer.setNeuronBias(n, bias);
                    }
                }
            }
            //TODO tornar x_i como sendo a saída da camada anterior
            //TODO guardar delta_w antes disso, para atualizar no final
            //update all neuron weights from all layers
            for (var l = 0; l < this.layers.length; l++) {
                var currentLayer = this.layers[l];
                // let X_considered: Array<number> ;
                // if (l == 0) {
                //     X_considered = X;
                // }
                for (var n = 0; n < currentLayer.getNNeurons(); n++) {
                    var neuronWeights = currentLayer.getNeuronWeights(n);
                    for (var p = 0; p < X.length; p++) {
                        var x_row = X[p];
                        for (var i = 0; i < x_row.length; i++) {
                            var x_i = x_row[i];
                            var currentWeight = neuronWeights[i];
                            neuronWeights[i] = currentWeight -
                                (this.eta * currentLayer.getNeuronError(n) * x_i);
                        }
                    }
                    currentLayer.setNeuronWeights(n, neuronWeights);
                }
            }
        }
    };
    MLP.prototype.predict = function (x_row) {
        return this.propagate(x_row);
    };
    return MLP;
}());
var Layer = /** @class */ (function () {
    function Layer(n_inputs, n_neurons, activationFunction) {
        this.weights = new Array();
        this.biases = new Array();
        this.errors = new Array();
        this.deltas = new Array();
        this.activations = new Array();
        this.delta_weights = new Array();
        this.n_inputs = n_inputs;
        this.n_neurons = n_neurons;
        this.activationFunction = activationFunction;
        this.eta = 0.01;
        this.initNeurons();
    }
    Layer.prototype.initNeurons = function () {
        this.weights = new Array();
        this.delta_weights = new Array();
        this.errors = new Array();
        this.biases = new Array();
        this.deltas = new Array();
        this.activations = new Array();
        for (var neuron = 0; neuron < this.n_neurons; neuron++) {
            this.weights[neuron] = new Array();
            this.delta_weights[neuron] = new Array();
            for (var input = 0; input < this.n_inputs; input++) {
                this.weights[neuron].push(Math.random());
            }
            this.biases[neuron] = 1;
            this.errors[neuron] = 0;
            this.deltas[neuron] = 0;
        }
    };
    Layer.prototype.getNInputs = function () {
        return this.n_inputs;
    };
    Layer.prototype.getNNeurons = function () {
        return this.n_neurons;
    };
    Layer.prototype.getNeuronWeights = function (neuron_idx) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        return this.weights[neuron_idx];
    };
    Layer.prototype.getNeuronError = function (neuron_idx) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        return this.errors[neuron_idx];
    };
    Layer.prototype.getNeuronDelta = function (neuron_idx) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        return this.deltas[neuron_idx];
    };
    Layer.prototype.getNeuronDeltaWeights = function (neuron_idx) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        return this.delta_weights[neuron_idx];
    };
    Layer.prototype.getNeuronBias = function (neuron_idx) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        return this.biases[neuron_idx];
    };
    Layer.prototype.getActivations = function () {
        return this.activations;
    };
    Layer.prototype.setEta = function (eta) {
        this.eta = eta;
    };
    Layer.prototype.setActivations = function (activations) {
        this.activations = activations;
    };
    Layer.prototype.setNeuronError = function (neuron_idx, value) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        this.errors[neuron_idx] = value;
    };
    Layer.prototype.setNeuronDelta = function (neuron_idx, value) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        this.deltas[neuron_idx] = value;
    };
    Layer.prototype.setNeuronBias = function (neuron_idx, value) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        this.biases[neuron_idx] = value;
    };
    Layer.prototype.setNeuronWeights = function (neuron_idx, weights) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        this.weights[neuron_idx] = weights;
    };
    Layer.prototype.setNeuronDeltaWeights = function (neuron_idx, weights) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        this.delta_weights[neuron_idx] = weights;
    };
    Layer.prototype.calculateLayerActivation = function (x_row) {
        var activations = new Array();
        for (var neuron = 0; neuron < this.n_neurons; neuron++) {
            var activation = this.calculateNeuronActivation(neuron, x_row);
            activations.push(activation);
        }
        return activations;
    };
    Layer.prototype.calculateNeuronActivation = function (neuron_idx, x_row) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        var prediction = 0;
        for (var i in x_row) {
            prediction += ((x_row[i] * this.weights[neuron_idx][i] * this.eta)
                + this.biases[neuron_idx]);
        }
        var activation = this.activationFunction.apply(prediction);
        if (isNaN(activation)) {
            throw new Exception('Resulted in NaN');
        }
        return activation;
    };
    Layer.prototype.calculateNeuronBias = function (neuron_idx, error) {
        var bias = this.biases[neuron_idx] + error * this.eta;
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        if (isNaN(bias)) {
            console.log(this.n_neurons);
            console.log(neuron_idx, this.biases[neuron_idx], error, this.eta);
            throw new Exception('Resulted in NaN');
        }
        return bias;
    };
    Layer.prototype.calculateNeuronError = function (neuron_idx, x_row, y_d) {
        var error = y_d - this.calculateNeuronActivation(neuron_idx, x_row);
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        if (isNaN(error)) {
            throw new Exception('Resulted in NaN');
        }
        return error;
    };
    Layer.prototype.calculateNeuronDelta = function (neuron_idx, y_pred) {
        var delta = this.errors[neuron_idx] * this.activationFunction.derivate(y_pred);
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        if (isNaN(delta)) {
            throw new Exception('Resulted in NaN');
        }
        return delta;
    };
    Layer.prototype.updateNeuronWeights = function (neuron_idx, x_row, error) {
        if (neuron_idx > this.getNNeurons() - 1 && neuron_idx < 0) {
            throw new Exception('Neuron index out of bounds');
        }
        for (var i = 0; i < this.n_inputs; i++) {
            this.weights[neuron_idx][i] = this.weights[neuron_idx][i]
                + (error * x_row[i] * this.eta);
        }
    };
    return Layer;
}());
var layer1 = new Layer(2, 3, new Sigmoid);
var layer2 = new Layer(3, 3, new Sigmoid);
var layer3 = new Layer(3, 2, new Sigmoid);
var mlp = new MLP([layer1, layer2, layer3], 0.01, 10);
var X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
var y = [[0, 0, 0, 1], [0, 1, 1, 1]];
mlp.train(X, y);
console.log(mlp.predict([0, 0]));
console.log(mlp.predict([0, 1]));
console.log(mlp.predict([1, 0]));
console.log(mlp.predict([1, 1]));
