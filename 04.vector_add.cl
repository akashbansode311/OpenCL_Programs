__kernel void add_vectors(__global int* A, __global int* B, __global int* C, int N) {
    int id = get_global_id(0);
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}
