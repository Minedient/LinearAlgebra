#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define tileSize 16
#include <clblast_c.h>
kernel void multiply(const int M, const int N, const int K,
 const global double *A, const global double *B, global double *C){

    const int localRow = get_local_id(0);           // local row ID (max: tileSize)
    const int localColumn = get_local_id(1);        // local column ID (max: tileSize)
    const int globalRow = get_global_id(0);         // row ID of C (0..M)
    const int globalColumn = get_global_id(1);      // column ID of C (0..N)

    //printf("localRow: %i\t localColumn: %i\t globalRow: %i\t globalColumn: %i\n", localRow, localColumn, globalRow, globalColumn);
    // Local memory
    __local double aSubMatrix[tileSize][tileSize];
    __local double bSubMatrix[tileSize][tileSize];

    // Sum register
    double sum = 0.0;

    const int numOfTiles = ceil(K * 1.0 / tileSize);

    // Loop through all tiles
    for(int t=0; t < numOfTiles; t++){
        const int tiledRow = tileSize * t + localRow;
        const int tiledColumn = tileSize * t + localColumn;
        aSubMatrix[localRow][localColumn] = A[globalRow * M + tiledColumn];
        bSubMatrix[localRow][localColumn] = B[tiledRow * N + globalColumn];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k=0; k < tileSize; k++){
            sum += aSubMatrix[localRow][k] * bSubMatrix[k][localColumn];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalRow * N + globalColumn] = sum;
}