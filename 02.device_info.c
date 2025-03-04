#include <CL/cl.h>
#include <stdio.h>

int main() {
    cl_uint num_platforms;
    
    // Get the number of OpenCL platforms
    if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) {
        printf("Failed to get OpenCL platforms.\n");
        return 1;
    }
    
    if (num_platforms == 0) {
        printf("No OpenCL platforms found.\n");
        return 1;
    }

    printf("OpenCL Platforms Found: %u\n", num_platforms);

    // Get platform IDs
    cl_platform_id platforms[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("Platform %d: %s\n", i + 1, platform_name);

        // Get number of devices
        cl_uint num_devices;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices) != CL_SUCCESS) {
            printf("  Failed to get OpenCL devices for this platform.\n");
            continue;
        }

        printf("  Devices Found: %u\n", num_devices);
        
        // Get device IDs
        cl_device_id devices[num_devices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[1024];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("    Device %d: %s\n", j + 1, device_name);
        }
    }

    return 0;
}
