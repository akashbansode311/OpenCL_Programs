__kernel void fp64Kernel(__global double *a, 
                         __global double *b, 
                         __global double *c, 
                         __global double *d, 
                         const int N) {
    int global_id = get_global_id(0);   // Unique thread ID (global)
    int local_id = get_local_id(0);     // Local thread ID within the work-group
    int group_id = get_group_id(0);     // Work-group ID

    if (global_id < N) {
        double mul_result = a[global_id] * b[global_id];
        d[global_id] = mul_result + c[global_id];

        // Print execution details for debugging (remove for performance)
        printf("Global ID: %d, Local ID: %d, Work-group ID: %d\n", global_id, local_id, group_id);
    }
}
