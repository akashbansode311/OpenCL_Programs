__kernel void add_vectors(__global float* A, __global float* B, __global float* C, int N) {
    int id = get_global_id(0);
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}
