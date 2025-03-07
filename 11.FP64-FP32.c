#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

#define N 1024

int main() {
    printf("Size of double (FP64): %lu bytes\n", sizeof(double));
    printf("Size of float (FP32): %lu bytes\n", sizeof(float));
    printf("Size of int: %lu bytes\n", sizeof(int));

    double *h_a_fp64 = (double*)malloc(N * sizeof(double));
    double *h_b_fp64 = (double*)malloc(N * sizeof(double));
    double *h_c_fp64 = (double*)malloc(N * sizeof(double));
    double *h_d_fp64 = (double*)malloc(N * sizeof(double));

    float *h_a_fp32 = (float*)malloc(N * sizeof(float));
    float *h_b_fp32 = (float*)malloc(N * sizeof(float));
    float *h_c_fp32 = (float*)malloc(N * sizeof(float));
    float *h_d_fp32 = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a_fp64[i] = i * 1.0;
        h_b_fp64[i] = i * 2.0;
        h_c_fp64[i] = i * 3.0;
        h_a_fp32[i] = (float)i * 4.0f;
        h_b_fp32[i] = (float)i * 5.0f;
        h_c_fp32[i] = (float)i * 6.0f;
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_fp64, kernel_fp32;
    cl_mem d_a_fp64, d_b_fp64, d_c_fp64, d_d_fp64;
    cl_mem d_a_fp32, d_b_fp32, d_c_fp32, d_d_fp32;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    FILE *fp = fopen("11.FP64-FP32.cl", "r");
    fseek(fp, 0, SEEK_END);
    size_t kernel_size = ftell(fp);
    rewind(fp);
    char *kernel_source = (char*)malloc(kernel_size + 1);
    fread(kernel_source, 1, kernel_size, fp);
    kernel_source[kernel_size] = '\0';
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    free(kernel_source);

    kernel_fp64 = clCreateKernel(program, "fp64Kernel", &err);
    kernel_fp32 = clCreateKernel(program, "fp32Kernel", &err);

    d_a_fp64 = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(double), NULL, &err);
    d_b_fp64 = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(double), NULL, &err);
    d_c_fp64 = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(double), NULL, &err);
    d_d_fp64 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(double), NULL, &err);
    d_a_fp32 = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &err);
    d_b_fp32 = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &err);
    d_c_fp32 = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &err);
    d_d_fp32 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);

    clEnqueueWriteBuffer(queue, d_a_fp64, CL_TRUE, 0, N * sizeof(double), h_a_fp64, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b_fp64, CL_TRUE, 0, N * sizeof(double), h_b_fp64, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_c_fp64, CL_TRUE, 0, N * sizeof(double), h_c_fp64, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_a_fp32, CL_TRUE, 0, N * sizeof(float), h_a_fp32, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b_fp32, CL_TRUE, 0, N * sizeof(float), h_b_fp32, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_c_fp32, CL_TRUE, 0, N * sizeof(float), h_c_fp32, 0, NULL, NULL);

    clSetKernelArg(kernel_fp64, 0, sizeof(cl_mem), &d_a_fp64);
    clSetKernelArg(kernel_fp64, 1, sizeof(cl_mem), &d_b_fp64);
    clSetKernelArg(kernel_fp64, 2, sizeof(cl_mem), &d_c_fp64);
    clSetKernelArg(kernel_fp64, 3, sizeof(cl_mem), &d_d_fp64);
    clSetKernelArg(kernel_fp32, 0, sizeof(cl_mem), &d_a_fp32);
    clSetKernelArg(kernel_fp32, 1, sizeof(cl_mem), &d_b_fp32);
    clSetKernelArg(kernel_fp32, 2, sizeof(cl_mem), &d_c_fp32);
    clSetKernelArg(kernel_fp32, 3, sizeof(cl_mem), &d_d_fp32);

    size_t global_size = N;

    clock_t start_fp64 = clock();
    clEnqueueNDRangeKernel(queue, kernel_fp64, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    clock_t end_fp64 = clock();
    double time_fp64 = ((double)(end_fp64 - start_fp64)) / CLOCKS_PER_SEC * 1000;

    clock_t start_fp32 = clock();
    clEnqueueNDRangeKernel(queue, kernel_fp32, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    clock_t end_fp32 = clock();
    double time_fp32 = ((double)(end_fp32 - start_fp32)) / CLOCKS_PER_SEC * 1000;

    clEnqueueReadBuffer(queue, d_d_fp64, CL_TRUE, 0, N * sizeof(double), h_d_fp64, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_d_fp32, CL_TRUE, 0, N * sizeof(float), h_d_fp32, 0, NULL, NULL);


    printf("FP64 Multiplication and Addition Time: %.2f ms\n", time_fp64);
    printf("FP32 Multiplication and Addition Time: %.2f ms\n", time_fp32);

    clReleaseMemObject(d_a_fp64);
    clReleaseMemObject(d_b_fp64);
    clReleaseMemObject(d_c_fp64);
    clReleaseMemObject(d_d_fp64);
    clReleaseMemObject(d_a_fp32);
    clReleaseMemObject(d_b_fp32);
    clReleaseMemObject(d_c_fp32);
    clReleaseMemObject(d_d_fp32);
    clReleaseKernel(kernel_fp64);
    clReleaseKernel(kernel_fp32);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_a_fp64);
    free(h_b_fp64);
    free(h_c_fp64);
    free(h_d_fp64);
    free(h_a_fp32);
    free(h_b_fp32);
    free(h_c_fp32);
    free(h_d_fp32);

    return 0;
}
