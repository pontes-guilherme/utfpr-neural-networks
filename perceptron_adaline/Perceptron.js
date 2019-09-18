export default class Perceptron {

    constructor(eta = 0.1, epochs = 1000) {
        this._eta = eta;
        this._epochs = epochs;
        this._weights = [];
        this._outputNeurons = 1;
        this._bias = 0
    }

    rand(min, max) {
        return Math.random() * (max - min) + min;
    }

    setOutputNeurons(size) {
        this._outputNeurons = size;
    }

    getOutputNeurons() {
        return this._outputNeurons;
    }

    getBias() {
        return this._bias;
    }

    setBias(bias) {
        this._bias = bias;
    }

    /**
     * Inicia a lista de pesos
     * 
     * @param int size Tamanho da lista de pesos para cada padrão de treinamento
     */
    initWeights(size) {
        for (let i = 0; i < this.getOutputNeurons(); i++) {
            this._weights[i] = [];

            for (let s = 0; s < size; s++) {
                this._weights[i][s] = this.rand(-1, 1);
            }
        }
    }

    fit(inputs, outputs) {
        this.setOutputNeurons(outputs[0].length);
        this.initWeights(inputs[0].length);
        this.setBias(1);

        for (let epoch = 0; epoch < this._epochs; epoch++) {
            let sum_of_errors = 0;

            for (let p = 0; p < inputs.length; p++) {
                let inputRow = inputs[p];

                let y_pred = this.predict(inputRow);

                for (let j = 0; j < this.getOutputNeurons(); j++) {
                    let error = outputs[p][j] - y_pred[j];
                    this.setBias(this.getBias() + this._eta*error);

                    // sum_of_errors += error;
                    sum_of_errors += Math.abs(error);

                    for (let i = 0; i < inputRow.length; i++) {
                        let wji = this._weights[j][i];
                        this._weights[j][i] = wji + (this._eta*error*inputRow[i])
                    }
                }

            }

            if (sum_of_errors == 0) 
                return;
        }
    }

    predict(inputRow) {
        let predictions = [];
        
        for (let n = 0; n < this.getOutputNeurons(); n++) {
            let sum = this.getBias();
            
            for (let i = 0; i < inputRow.length; i++) {
                sum += inputRow[i] * this._weights[n][i];
            }

            predictions[n] = sum;
        }

        return this.activate(predictions);
    }

    activate(prediction) {
        let activations = [];

        for (let n = 0; n < this.getOutputNeurons(); n++) {
            activations[n] = prediction[n] > 0 ? 1 : 0;
        }

        return activations;
    }
}

