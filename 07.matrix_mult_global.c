#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MATRIX_SIZE 25000  // Define matrix size (25000x25000)

// OpenCL kernel file
#define KERNEL_FILE "07.matrix_mult_global.cl"

int main() {
    int width = MATRIX_SIZE;
    size_t size = width * width * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;  // Fill A with 1.0
        h_B[i] = 2.0f;  // Fill B with 2.0
    }

    // Load OpenCL platform & device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context & queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    // Create OpenCL buffers
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_A, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_B, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);

    // Load kernel source
    FILE *fp = fopen(KERNEL_FILE, "r");
    fseek(fp, 0, SEEK_END);
    size_t kernel_size = ftell(fp);
    rewind(fp);
    char *kernel_source = (char*)malloc(kernel_size + 1);
    fread(kernel_source, 1, kernel_size, fp);
    kernel_source[kernel_size] = '\0';
    fclose(fp);

    // Build the kernel
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, NULL);
    cl_int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &width);

    // Define work sizes
    size_t local_size[] = { 16, 16 };  // Workgroup size
    size_t global_size[] = { (width + local_size[0] - 1) / local_size[0] * local_size[0], 
                             (width + local_size[1] - 1) / local_size[1] * local_size[1] };

    // Run kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error during operation 'clEnqueueNDRangeKernel': %d\n", err);
        return -1;
    }

    // Copy results back to host
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size, h_C, 0, NULL, NULL);

    // Print sample result
    printf("Sample result: C[0][0] = %f\n", h_C[0]);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(kernel_source);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
