export namespace Constants {
    export const K: Number = 3;
    export const ITERATIONS: Number = 1000;
}

export namespace Interfaces {
    export interface IClustering {
        cluster(data: Number[][]): Object[];
        getNumberOfClusters(): Number;
    }

    export interface ICentroid {
        center: Number[];
    }
}