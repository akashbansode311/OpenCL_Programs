__kernel void hello_kernel() {
    int id = get_global_id(0);
    printf("Hello, World from GPU! (Thread ID: %d)\n", id);
}
