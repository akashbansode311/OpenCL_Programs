#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define DATA_SIZE 10  // Number of work items (threads)

int main() {
    // Step 1: Print message from CPU
    printf("Hello, World from CPU!\n");

    // Step 2: Get Platform & Device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    // Step 3: Create OpenCL Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Step 4: Create OpenCL Command Queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    // Step 5: Read Kernel Code from File
    FILE *fp = fopen("01.hello_cpu_gpu.cl", "r");
    fseek(fp, 0, SEEK_END);
    size_t kernelSize = ftell(fp);
    rewind(fp);
    char *kernelSource = (char*)malloc(kernelSize + 1);
    fread(kernelSource, 1, kernelSize, fp);
    kernelSource[kernelSize] = '\0';
    fclose(fp);

    // Step 6: Create OpenCL Program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernelSize, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Step 7: Create Kernel
    cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);

    // Step 8: Execute the Kernel
    size_t globalSize = DATA_SIZE;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Step 9: Wait for completion
    clFinish(queue);

    // Cleanup
    free(kernelSource);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
