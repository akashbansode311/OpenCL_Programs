__kernel void floatKernel(__global float *a, __global float *b, __global float *c, __global float *d, int N) {
    int globalThreadId = get_global_id(0);
    
    if (globalThreadId < N) {
        // Perform multiplication
        float mul_result = a[globalThreadId] * b[globalThreadId];
        // Perform addition
        d[globalThreadId] = mul_result + c[globalThreadId];
    }
}
