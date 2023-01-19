#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double activationMode(const double value, const int MODE){

    switch(MODE){
        case 1: // Sigmoid
            return 1 / (1 + exp(-value));
        case 2: // Step
            return (value >= 0) ? 1 : 0;
        case 3: // Tanh
            return tanh(value);
        case 4: // SoftPlus
            return log(1 + exp(value));
        case 0: // LeakyReLU
        default:
            return (value >= 0) ? value : 0.01 * value;
    }

}

kernel void forwardPass(const int M, const int N, const int K, const global double *weightMatrix, const global double *inputsMatrix, global double *biasMatrix, global double *resultsMatrix, const int MODE){

    const int row = get_global_id(0);
    const int column = get_global_id(1);

    // simple loop over the common side of two matrix
    double temp = 0.0;
    for (int i=0; i<K; i++){
        temp += weightMatrix[row*M + i] * inputsMatrix[i*N + column];
    }

    // set the result
    temp += biasMatrix[row*N + column];
    resultsMatrix[row*N + column] = activationMode(temp, MODE);
}