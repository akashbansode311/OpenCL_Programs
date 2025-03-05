__kernel void add_vectors(__global double *A, __global double *B, __global double *C, const int N) {
    int id = get_global_id(0);
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}
