__kernel void int32Kernel(__global int *a, __global int *b, __global int *c, __global int *d, int n) {
    int global_id = get_global_id(0);  // Get the global thread index

    if (global_id < n) {
        int mul_result = a[global_id] * b[global_id];  // Multiply
        d[global_id] = mul_result + c[global_id];      // Add
    }
}
