#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void multiply(const int M, const int N, const int K,
 const global double *A, const global double *B, global double *C){
    // global is a must!!!                       no MF const here!

    const int row = get_global_id(0);
    const int column = get_global_id(1);


    // simple loop over the common side of two matrix
    double temp = 0.0;
    for (int i=0; i<K; i++){
        temp += A[row*M + i] * B[i*N + column];
    }

    // set the result
    C[row*N + column] = temp;
}