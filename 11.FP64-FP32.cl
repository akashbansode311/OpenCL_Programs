__kernel void fp64Kernel(__global double *a, __global double *b, __global double *c, __global double *d) {
    int id = get_global_id(0);
    d[id] = a[id] * b[id] + c[id];
}

__kernel void fp32Kernel(__global float *a, __global float *b, __global float *c, __global float *d) {
    int id = get_global_id(0);
    d[id] = a[id] * b[id] + c[id];
}
