var NORM_START = 0.1;
var NORM_END = 0.9;
function norm(value, min, max, ra, rb) {
    return (((ra - rb) * (value - min)) / (max - min)) + rb;
}
var flatten = function flatten(list) {
    return list.reduce(function (a, b) {
        return a.concat(Array.isArray(b) ? flatten(b) : b);
    }, []);
};
var min = function (l) {
    return l.reduce(function (acc, item) {
        return item < acc ? item : acc;
    }, Infinity);
};
var max = function (l) {
    return l.reduce(function (acc, item) {
        return item > acc ? item : acc;
    }, -Infinity);
};
var Sigmoid = /** @class */ (function () {
    function Sigmoid() {
    }
    Sigmoid.prototype.apply = function (value) {
        var result = (1 / (1 + value));
        return result;
    };
    Sigmoid.prototype.derivate = function (value) {
        var result = value * (1 - value);
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
        /* NORMALIZAÇÂO */
        var X_NORM = Array();
        var flat = flatten(X);
        var min_val = min(flat);
        var max_val = max(flat);
        for (var p = 0; p < X.length; p++) {
            var x_row = X[p];
            X_NORM[p] = Array();
            for (var i in x_row) {
                var x = x_row[i];
                X_NORM[p][i] = norm(x, min_val, max_val, NORM_START, NORM_END);
            }
        }
        for (var epoch = 0; epoch < this.epochs; epoch++) {
            //limpa os delta weights no início de cada época
            for (var l = this.layers.length - 1; l >= 0; l--) {
                var currentLayer = this.layers[l];
                currentLayer.clearDeltaWeights();
            }
            var lastLayer = this.getLastLayer();
            var penultimateLayer = this.layers[this.layers.length - 2];
            //para cada padrão de treinamento
            for (var p = 0; p < X_NORM.length; p++) {
                var x_row = X_NORM[p];
                var y_pred = this.propagate(x_row);
                //iterar por cada neurônio da última camada
                for (var n = 0; n < lastLayer.getNNeurons(); n++) {
                    var y_desired_neuron = y[n];
                    var y_desired = y_desired_neuron[p];
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
                    var neuron_delta_weights = lastLayer.getNeuronDeltaWeights(n);
                    //setar delta_weight baseado na saída da última camada
                    var delta_weight = [];
                    var penultimate_layer_x_row = penultimateLayer.getActivations();
                    for (var j in penultimate_layer_x_row) {
                        var x_from_previous_layer = penultimate_layer_x_row[j];
                        var previous_delta = neuron_delta_weights.length ? neuron_delta_weights[j] : 0;
                        var newDeltaWeight = (previous_delta + (this.eta * lastLayer.getNeuronError(n) * x_from_previous_layer));
                        delta_weight.push(newDeltaWeight);
                    }
                    lastLayer.setNeuronDeltaWeights(n, delta_weight);
                }
                for (var l = this.layers.length - 2; l >= 0; l--) {
                    var nextLayer = this.layers[l + 1];
                    var currentLayer = this.layers[l];
                    var x_row_considered = void 0;
                    if (l > 0) {
                        var previousLayer = this.layers[l - 1];
                        x_row_considered = previousLayer.getActivations();
                    }
                    else {
                        x_row_considered = x_row;
                    }
                    //iterar por cada neurônio da camada
                    for (var n = 0; n < currentLayer.getNNeurons(); n++) {
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
                        var neuron_delta_weights = currentLayer.getNeuronDeltaWeights(n);
                        //setar delta_weight baseado na saída da última camada
                        var delta_weight = [];
                        for (var j in x_row_considered) {
                            var x_from_previous_layer = x_row_considered[j];
                            var previous_delta = neuron_delta_weights.length ? neuron_delta_weights[j] : 0;
                            var newDeltaWeight = (previous_delta + (this.eta * currentLayer.getNeuronError(n) * x_from_previous_layer));
                            delta_weight.push(newDeltaWeight);
                        }
                        currentLayer.setNeuronDeltaWeights(n, delta_weight);
                    }
                }
            }
            //update all neuron weights from all layers
            for (var l = 0; l < this.layers.length; l++) {
                var currentLayer = this.layers[l];
                for (var n = 0; n < currentLayer.getNNeurons(); n++) {
                    var neuronWeights = currentLayer.getNeuronWeights(n);
                    var neuronDeltaWeights = currentLayer.getNeuronDeltaWeights(n);
                    for (var i in neuronDeltaWeights) {
                        neuronWeights[i] += neuronDeltaWeights[i];
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
                this.weights[neuron].push(Math.random() * 2 - 1);
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
        return this.weights[neuron_idx];
    };
    Layer.prototype.getNeuronError = function (neuron_idx) {
        return this.errors[neuron_idx];
    };
    Layer.prototype.getNeuronDelta = function (neuron_idx) {
        return this.deltas[neuron_idx];
    };
    Layer.prototype.getNeuronDeltaWeights = function (neuron_idx) {
        return this.delta_weights[neuron_idx];
    };
    Layer.prototype.getNeuronBias = function (neuron_idx) {
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
        this.errors[neuron_idx] = value;
    };
    Layer.prototype.setNeuronDelta = function (neuron_idx, value) {
        this.deltas[neuron_idx] = value;
    };
    Layer.prototype.setNeuronBias = function (neuron_idx, value) {
        this.biases[neuron_idx] = value;
    };
    Layer.prototype.setNeuronWeights = function (neuron_idx, weights) {
        this.weights[neuron_idx] = weights;
    };
    Layer.prototype.setNeuronDeltaWeights = function (neuron_idx, weights) {
        this.delta_weights[neuron_idx] = weights;
    };
    Layer.prototype.clearDeltaWeights = function () {
        this.delta_weights = new Array();
        for (var neuron = 0; neuron < this.n_neurons; neuron++) {
            this.delta_weights[neuron] = new Array();
        }
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
        var prediction = 0;
        for (var i in x_row) {
            prediction += (x_row[i] * this.weights[neuron_idx][i] * this.eta);
        }
        prediction += this.biases[neuron_idx];
        var activation = this.activationFunction.apply(prediction);
        return activation;
    };
    Layer.prototype.calculateNeuronBias = function (neuron_idx, error) {
        var bias = this.biases[neuron_idx] + error * this.eta;
        return bias;
    };
    Layer.prototype.calculateNeuronDelta = function (neuron_idx, y_pred) {
        var delta = this.errors[neuron_idx] * this.activationFunction.derivate(y_pred);
        return delta;
    };
    return Layer;
}());
var layer1 = new Layer(2, 3, new Sigmoid);
var layer2 = new Layer(3, 3, new Sigmoid);
var layer3 = new Layer(3, 2, new Sigmoid);
var mlp = new MLP([layer1, layer2, layer3], 0.01, 1000);
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
