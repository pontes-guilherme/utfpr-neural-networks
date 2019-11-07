export namespace Constants {
    export const K: number = 3;
    export const ITERATIONS: number = 1000;
}

export namespace Interfaces {
    export interface IClustering {
        cluster(data: number[][]): Object[];
        getNumberOfClusters(): number;
    }

    // export interface ICentroid {
    //     center: Number[];
    // }
}