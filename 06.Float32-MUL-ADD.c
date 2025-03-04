#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define ARRAY_SIZE (5120 * 10000)  // Array size
#define KERNEL_FILE "06.Float32-MUL-ADD.cl"

int main() {
    const int N = ARRAY_SIZE;  // Define N as a constant
    const size_t size_fp32 = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size_fp32);
    float *h_b = (float*)malloc(size_fp32);
    float *h_c = (float*)malloc(size_fp32);
    float *h_d = (float*)malloc(size_fp32);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i * 1.0f;
        h_b[i] = (float)i * 2.0f;
        h_c[i] = (float)i * 3.0f;
    }

    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_a, d_b, d_c, d_d;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    // Allocate device memory
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, size_fp32, NULL, &err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, size_fp32, NULL, &err);
    d_c = clCreateBuffer(context, CL_MEM_READ_ONLY, size_fp32, NULL, &err);
    d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_fp32, NULL, &err);

    // Copy data to device
    clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, size_fp32, h_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, size_fp32, h_b, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_c, CL_TRUE, 0, size_fp32, h_c, 0, NULL, NULL);

    // Read kernel source
    FILE *fp = fopen(KERNEL_FILE, "r");
    fseek(fp, 0, SEEK_END);
    size_t kernel_size = ftell(fp);
    rewind(fp);
    char *kernel_source = (char*)malloc(kernel_size + 1);
    fread(kernel_source, 1, kernel_size, fp);
    kernel_source[kernel_size] = '\0';
    fclose(fp);

    // Create and build OpenCL program
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_size, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "fp32Kernel", &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_d);
    clSetKernelArg(kernel, 4, sizeof(int), &N);

    // Launch kernel
    size_t globalSize = N;
    size_t localSize = 128;  // Can be tuned
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);

    // Copy results back to host
    clEnqueueReadBuffer(queue, d_d, CL_TRUE, 0, size_fp32, h_d, 0, NULL, NULL);

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

    printf("OpenCL FP32 kernel execution completed.\n");
    return 0;
}
