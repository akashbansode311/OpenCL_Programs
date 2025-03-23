__kernel void matrix_multiply(__global float* A, __global float* B, __global float* C, int n) {
    int row = get_global_id(0); // Row index
    int col = get_global_id(1); // Column index

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
