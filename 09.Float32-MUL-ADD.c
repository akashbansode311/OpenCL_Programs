#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define KERNEL_FILE "09.Float32-MUL-ADD.cl"
#define ARRAY_SIZE (5120 * 10000)  // Array size

int main() {
    const int N = ARRAY_SIZE;  // Define N as a constant variable

    float *h_a, *h_b, *h_c, *h_d; // Host memory
    cl_mem d_a, d_b, d_c, d_d;  // Device memory
    cl_int err;

    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_d = (float*)malloc(size);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = 2.0f * (float)i;
        h_c[i] = 3.0f * (float)i;
    }

    // Get platform and device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Create command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    // Allocate device memory
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
    d_c = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
    d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);

    // Copy data to device
    clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, size, h_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, size, h_b, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_c, CL_TRUE, 0, size, h_c, 0, NULL, NULL);

    // Load and compile the kernel
    FILE *fp = fopen(KERNEL_FILE, "r");
    fseek(fp, 0, SEEK_END);
    size_t kernel_size = ftell(fp);
    rewind(fp);
    char *kernel_source = (char*)malloc(kernel_size + 1);
    fread(kernel_source, 1, kernel_size, fp);
    kernel_source[kernel_size] = '\0';
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);

    // Build program
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "floatKernel", &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_d);
    clSetKernelArg(kernel, 4, sizeof(int), &N);

    // Execute kernel
    size_t global_size = N;
    size_t local_size = 128;  // Work-group size
    cl_event event;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);

    // Wait for kernel to finish
    clWaitForEvents(1, &event);

    // Read result from device to host
    clEnqueueReadBuffer(queue, d_d, CL_TRUE, 0, size, h_d, 0, NULL, NULL);

    // Verify output by printing the first 10 elements
    for (int i = 0; i < 10; i++) {
    printf("h_d[%d] = %f\n", i, h_d[i]);
    }
    // Cleanup
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseMemObject(d_d);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(kernel_source);

    printf("Execution completed successfully!\n");

    return 0;
}
