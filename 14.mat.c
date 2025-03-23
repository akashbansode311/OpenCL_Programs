#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define N 2000 // Matrix size (N x N)

// Function to load the OpenCL kernel from file
char* load_kernel_source(const char* filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Failed to load kernel.\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    char *source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

int main() {
    int size = N * N;
    float *A, *B, *C;

    // Allocate memory for matrices
    A = (float*)malloc(size * sizeof(float));
    B = (float*)malloc(size * sizeof(float));
    C = (float*)malloc(size * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < size; i++) {
        A[i] = 1; // Example values
        B[i] = 2;
    }

    // Load kernel source code
    char *kernelSource = load_kernel_source("14.mat.cl");

    // Get platform & device information
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

    // Create buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), NULL, NULL);

    // Copy data to GPU
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, size * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, size * sizeof(float), B, 0, NULL, NULL);

    // Create OpenCL program from source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", NULL);

    // Set kernel arguments
    int n = N; 
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Define global and local work sizes
    size_t globalSize[2] = {N, N};
    size_t localSize[2] = {16, 16}; // Adjust based on GPU capabilities

    // Execute kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);

    // Read result from GPU
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, size * sizeof(float), C, 0, NULL, NULL);

    // Print a sample output
    printf("C[0][0] = %f\n", C[0]);

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(kernelSource);

    return 0;
}
