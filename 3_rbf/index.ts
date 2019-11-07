import { Constants, Interfaces } from './imports';

const clusterMaker = require('clusters');

const N_CLUSTERS = Constants.K;
const ITERATIONS = Constants.ITERATIONS;


class Clustering implements Interfaces.IClustering {
    private number_of_clusters: number;
    private number_of_iterations: number;
    private clustering_algorithm;

    constructor(clustering_algorithm, number_of_clusters: number, number_of_iterations: number) {
        this.number_of_clusters = number_of_clusters;
        this.number_of_iterations = number_of_iterations;

        this.clustering_algorithm = clustering_algorithm;

        this.clustering_algorithm.k(this.number_of_clusters);
        this.clustering_algorithm.iterations(this.number_of_iterations);
    }

    public cluster(data: number[][]): Object[] {
        this.clustering_algorithm.data(data);
        return this.clustering_algorithm.clusters();
    }

    public getNumberOfClusters(): number {
        return this.number_of_clusters;
    }
}

class RBF {
    private clustering: Interfaces.IClustering;
    private n_neurons: number;
    private eta: number;

    private centroids: number[][];
    private sigmas: number[][];
    private weights: number[][];

    constructor(
        clustering: Interfaces.IClustering,
        eta: number,
    ) {
        this.clustering = clustering;
        this.eta = eta;

        this.n_neurons = 0;
        this.centroids = [];
        this.sigmas = [];
    }

    public train(data: number[][]) {
        const clusters = this.clustering.cluster(data);
        this.n_neurons = this.clustering.getNumberOfClusters();

        //definiçao dos clusteres
        clusters.map(({ centroid, points }, neuron_idx) => {
            if (!this.centroids[neuron_idx])
                this.centroids[neuron_idx] = new Array();

            this.centroids[neuron_idx].push(centroid);

            let dispersions: number[] = [];

            points.map(point => {
                let sum: number = 0;

                for (let i = 0; i < point.length; i++) {
                    let centroid_i: number = centroid[i];
                    let point_i: number = point[i];

                    sum += ((centroid_i - point_i) ** 2);
                }

                let dispersion = Math.sqrt(sum);
                dispersions.push(dispersion);
            });

            let maxDispersion: number = Math.max(...dispersions);

            if (!this.sigmas[neuron_idx])
                this.sigmas[neuron_idx] = new Array();

            this.sigmas[neuron_idx].push(maxDispersion);
        });

        console.log(this.centroids, this.sigmas)

    }

    // public combine() {

    // }


}

new RBF(new Clustering(clusterMaker, N_CLUSTERS, ITERATIONS), 0.01).train([[1, 2, 1], [3, 4, 7], [5, 6, 2], [1, 1, 1], [1, 1, 2], [9, 99, 6]])