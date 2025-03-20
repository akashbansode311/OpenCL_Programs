#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define ARRSIZE 200  // Define vector size

int main() {
    int *B, *C;
    size_t bytes = ARRSIZE * sizeof(int);

    // Allocate memory for host arrays
    B = (int *)malloc(bytes);
    C = (int *)malloc(bytes);

    // Initialize input array
    for (int i = 0; i < ARRSIZE; i++) {
        B[i] = i;
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
    FILE *file = fopen("12.vec_mul.c", "r");
    if (!file) {
        printf("Error: Unable to open kernel file\n");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    size_t kernelSize = ftell(file);
    rewind(file);
    char *kernelSource = (char *)malloc(kernelSize + 1);
    fread(kernelSource, 1, kernelSize, file);
    kernelSource[kernelSize] = '\0';
    fclose(file);

    // Create and build OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, &kernelSize, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    free(kernelSource);

    // Create OpenCL buffers
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Copy data to device
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, bytes, B, 0, NULL, NULL);

    // Create and set up kernel
    cl_kernel kernel = clCreateKernel(program, "vecSquare", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_C);

    // Define execution configuration
    size_t globalSize = ARRSIZE;

    // Execute kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Copy result back to host
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

    // Print results
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, C[i]);  // Print only first 10 results
    }

    // Cleanup
    free(B); free(C);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\nSuccess! Squaring operation completed.\n");

    return 0;
}
