import { Constants, Interfaces } from './imports';

let clusterMaker = require('clusters');

const N_CLUSTERS = Constants.K;
const ITERATIONS = Constants.ITERATIONS;


class Clustering implements Interfaces.IClustering {
    private number_of_clusters: Number;
    private number_of_iterations: Number;
    private clustering_algorithm;

    constructor(clustering_algorithm, number_of_clusters: Number, number_of_iterations: Number) {
        this.number_of_clusters = number_of_clusters;
        this.number_of_iterations = number_of_iterations;

        this.clustering_algorithm = clustering_algorithm;

        this.clustering_algorithm.k(this.number_of_clusters);
        this.clustering_algorithm.iterations(this.number_of_iterations);
    }

    public cluster(data: Number[][]): Object[] {
        this.clustering_algorithm.data(data);
        return this.clustering_algorithm.clusters();
    }

    public getNumberOfClusters(): Number {
        return this.number_of_clusters;
    }
}

class RBF {
    private clustering: Interfaces.IClustering;
    private n_neurons: Number;
    private eta: Number;

    private centroids: Interfaces.ICentroid[];
    private sigmas: Number[];

    constructor(
        clustering: Interfaces.IClustering,
        eta: Number,
    ) {
        this.clustering = clustering;
        this.eta = eta;

        this.n_neurons = 0;
        this.centroids = [];
    }

    public train(data: Number[][]) {
        const clusters = this.clustering.cluster(data);
        this.n_neurons = this.clustering.getNumberOfClusters();

        
    }


}