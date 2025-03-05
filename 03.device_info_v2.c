#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

int main() {
    // Get the number of platforms
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);

    if (platformCount == 0) {
        printf("No OpenCL platforms found.\n");
        return 1;
    }

    // Get platform IDs
    cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    // Iterate through each platform
    for (cl_uint i = 0; i < platformCount; i++) {
        char platformName[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
        printf("\nOpenCL Platform %d: %s\n", i, platformName);

        // Get the number of devices in the platform
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        
        if (deviceCount == 0) {
            printf("  No OpenCL devices found on this platform.\n");
            continue;
        }

        // Get device IDs
        cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // Iterate through each device
        for (cl_uint j = 0; j < deviceCount; j++) {
            char deviceName[256];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            printf("\n  Device %d: \"%s\"\n", j, deviceName);

            cl_uint computeUnits;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
            printf("    Compute Units: %u\n", computeUnits);

            cl_ulong globalMem;
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, NULL);
            printf("    Global Memory: %lu bytes\n", globalMem);

            cl_ulong constantMem;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(constantMem), &constantMem, NULL);
            printf("    Constant Memory: %lu bytes\n", constantMem);

            cl_ulong localMem;
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
            printf("    Local Memory (per work-group): %lu bytes\n", localMem);

            cl_uint maxWorkItemDims;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDims), &maxWorkItemDims, NULL);
            printf("    Max Work Item Dimensions: %u\n", maxWorkItemDims);

            size_t maxWorkItemSizes[3];
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, NULL);
            printf("    Max Work Item Sizes: %zu x %zu x %zu\n", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            size_t maxWorkGroupSize;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
            printf("    Max Work Group Size: %zu\n", maxWorkGroupSize);

            cl_ulong maxMemAllocSize;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
            printf("    Max Memory Allocation: %lu bytes\n", maxMemAllocSize);

            cl_uint clockFrequency;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFrequency), &clockFrequency, NULL);
            printf("    Clock Frequency: %u MHz\n", clockFrequency);
        }

        free(devices);
    }

    free(platforms);
    return 0;
}
