#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define ARRAY_SIZE (5120 * 10000)  // Array size

int main() {
    const int N = ARRAY_SIZE;  // Define N as a constant variable
    size_t bytes = N * sizeof(int);

    // Allocate memory for host arrays
    int *A = (int*)malloc(bytes);
    int *B = (int*)malloc(bytes);
    int *C = (int*)malloc(bytes);

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        A[i] = 1;
        B[i] = 2;
    }

    // Get OpenCL platform and device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    // Read kernel source from file
    FILE *file = fopen("03.vector_add.cl", "r");
    if (!file) {
        printf("Error: Unable to open kernel file\n");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    size_t kernelSize = ftell(file);
    rewind(file);
    char *kernelSource = (char*)malloc(kernelSize + 1);
    fread(kernelSource, 1, kernelSize, file);
    kernelSource[kernelSize] = '\0';
    fclose(file);

    // Create program and build
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernelSize, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    free(kernelSource);

    // Create OpenCL buffers
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Copy data to device
    clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, bytes, B, 0, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "add_vectors", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);  // Corrected: N is now a variable

    // Define execution configuration
    size_t globalSize = N;
    size_t localSize = 128;

    // Execute kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);

    // Copy result back to host
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

    // Verify results
    for (int i = 0; i < N; i++) {
        if (C[i] != 3) {
            printf("Error at index %d: Expected 3, but got %d\n", i, C[i]);
            exit(1);
        }
    }

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N = %d\n", N);
    printf("---------------------------\n\n");

    // Cleanup
    free(A); free(B); free(C);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\nSuccess! Vector addition completed.\n");

    return 0;
}
