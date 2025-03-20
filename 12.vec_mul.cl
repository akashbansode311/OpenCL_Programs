__kernel void vecSquare(__global int* B, __global int* C) {
    int gid = get_global_id(0);
    C[gid] = B[gid] * B[gid];
}
