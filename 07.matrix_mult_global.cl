__kernel void matrixMultiply(__global float *A, __global float *B, __global float *C, int width) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < width && col < width) {
        float sum = 0.0;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}
