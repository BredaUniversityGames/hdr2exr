#include "converter.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Include the necessary libraries to load an HDR image
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Include the necessary libraries to save the radiance and irradiance images in EXR format
#include "tinyexr.h"

// Include Vulkan headers to use the Vulkan (C++) API
#include <vulkan/vulkan.hpp>

using namespace std;

// As is from tinyexr example on github
// See `examples/rgbe2exr/` for more details.
bool SaveEXR(const float* rgb, int width, int height, const char* outfilename) {

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    // Split RGBRGBRGB... into R, G and B layer
    for (int i = 0; i < width * height; i++) {
      images[0][i] = rgb[3*i+0];
      images[1][i] = rgb[3*i+1];
      images[2][i] = rgb[3*i+2];
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
      header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
      header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = NULL; // or nullptr in C++11 or later.
    int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
    if (ret != TINYEXR_SUCCESS) {
      fprintf(stderr, "Save EXR err: %s\n", err);
      FreeEXRErrorMessage(err); // free's buffer for an error message
      return ret;
    }
    printf("Saved exr file. [ %s ] \n", outfilename);

    free((void*)rgb);
    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}


converter::converter(std::string input_file)
{
    // We assume that the input file exists and is an HDR image

    // Load the HDR image    
    data = stbi_loadf(input_file.c_str(), &width, &height, &channels, 0);
    if (!data)
    {
        cout << "Error: The input file " << input_file << " is not a valid HDR image." << endl;
        return;
    }
}

converter::~converter()
{
    // Free the memory allocated for the HDR image
    stbi_image_free(data);
}

bool converter::process()
{
    //////////////////////////////////////////////////////////////////////////
    //// Vulkan initialization code
    //////////////////////////////////////////////////////////////////////////
    vk::ApplicationInfo appInfo = vk::ApplicationInfo()
        .setPApplicationName("Vulkan HDR to Radiance Converter")
        .setApplicationVersion(1)
        .setPEngineName("Vulkan Engine")
        .setEngineVersion(1)
        .setApiVersion(VK_API_VERSION_1_3);

    vk::InstanceCreateInfo instanceCreateInfo = vk::InstanceCreateInfo().setPApplicationInfo(&appInfo);

    vk::Instance instance;
    try
    {
        instance = vk::createInstance(instanceCreateInfo);
        cout << "Vulkan instance created successfully." << endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create Vulkan instance: " << e.what() << std::endl;
        return false;
    }

    auto physicalDevices = instance.enumeratePhysicalDevices();
    if (physicalDevices.empty())
    {
        std::cerr << "No Vulkan physical devices found." << std::endl;
        return false;
    }

    auto physicalDevice = physicalDevices[0];
    auto properties = physicalDevice.getProperties();
    cout << "Vulkan physical device found: " << properties.deviceName << endl;

    uint32_t computeQueueFamilyIndex = -1;
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute)
        {
            computeQueueFamilyIndex = i;
            break;
        }
    }

    if (computeQueueFamilyIndex == -1)
    {
        std::cerr << "No Vulkan compute queue family found." << std::endl;
        return false;
    }

    float queuePriorities[] = { 1.0f };

    vk::DeviceQueueCreateInfo queueCreateInfo = vk::DeviceQueueCreateInfo()
        .setQueueFamilyIndex(computeQueueFamilyIndex)
        .setQueueCount(1)
        .setPQueuePriorities(queuePriorities);

    vk::DeviceCreateInfo deviceCreateInfo = vk::DeviceCreateInfo()
        .setQueueCreateInfoCount(1)
        .setPQueueCreateInfos(&queueCreateInfo);

    vk::Device device;

    try
    {
        device = physicalDevice.createDevice(deviceCreateInfo);
        cout << "Vulkan device created successfully." << endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create Vulkan device: " << e.what() << std::endl;
        return false;
    }

    vk::Queue queue = device.getQueue(computeQueueFamilyIndex, 0);

    //////////////////////////////////////////////////////////////////////////
    //// Vulkan code to process the HDR image
    //////////////////////////////////////////////////////////////////////////

    // Create a buffer to store the HDR image
    vk::BufferCreateInfo bufferCreateInfo = vk::BufferCreateInfo()
        .setSize(width * height * channels * sizeof(float))
        .setUsage(vk::BufferUsageFlagBits::eStorageBuffer);

    vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

    // Allocate memory for the buffer
    vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo memoryAllocateInfo = vk::MemoryAllocateInfo()
        .setAllocationSize(memoryRequirements.size)
        .setMemoryTypeIndex(0);

    vk::DeviceMemory memory = device.allocateMemory(memoryAllocateInfo);

    // Bind the buffer to the memory
    device.bindBufferMemory(buffer, memory, 0);

    // Copy the HDR image to the buffer
    void* mappedMemory = device.mapMemory(memory, 0, VK_WHOLE_SIZE);
    memcpy(mappedMemory, data, width * height * channels * sizeof(float));
    device.unmapMemory(memory);

    //////////////////////////////////////////////////////////////////////////
    //// Vulkan code to process the HDR image
    //////////////////////////////////////////////////////////////////////////

    // The radiance and irradiance images will be calculated and stored as cube
    // maps in the following order: radiance positive x, radiance negative x,
    // radiance positive y, radiance negative y, radiance positive z, radiance
    // negative z, irradiance positive x, irradiance negative x, irradiance
    // positive y, irradiance negative y, irradiance positive z, irradiance negative z.
    // Per roughness level, the radiance and irradiance images will be stored in
    // different layers of the same cube map.

    // Calculate the number of roughness levels
    int numRoughnessLevels = log2(width) - 1;

    // Create a buffer to store the radiance and irradiance images
    vk::BufferCreateInfo cubeMapBufferCreateInfo = vk::BufferCreateInfo()
        .setSize(width * height * channels * numRoughnessLevels * 12 * sizeof(float))
        .setUsage(vk::BufferUsageFlagBits::eStorageBuffer);

    vk::Buffer cubeMapBuffer = device.createBuffer(cubeMapBufferCreateInfo);

    // Allocate memory for the buffer
    vk::MemoryRequirements cubeMapMemoryRequirements = device.getBufferMemoryRequirements(cubeMapBuffer);

    vk::MemoryAllocateInfo cubeMapMemoryAllocateInfo = vk::MemoryAllocateInfo()
        .setAllocationSize(cubeMapMemoryRequirements.size)
        .setMemoryTypeIndex(0);

    vk::DeviceMemory cubeMapMemory = device.allocateMemory(cubeMapMemoryAllocateInfo);

    // Bind the buffer to the memory
    device.bindBufferMemory(cubeMapBuffer, cubeMapMemory, 0);

    // Create a descriptor set layout
    vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding = vk::DescriptorSetLayoutBinding()
        .setBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo()
        .setBindingCount(1)
        .setPBindings(&descriptorSetLayoutBinding);

    vk::DescriptorSetLayout descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

    // Create a pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo()
        .setSetLayoutCount(1)
        .setPSetLayouts(&descriptorSetLayout);

    vk::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

    // Create a descriptor pool
    vk::DescriptorPoolSize descriptorPoolSize = vk::DescriptorPoolSize()
        .setType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(2);

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo()
        .setMaxSets(1)
        .setPoolSizeCount(1)
        .setPPoolSizes(&descriptorPoolSize);

    vk::DescriptorPool descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);

    // Create a descriptor set
    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo()
        .setDescriptorPool(descriptorPool)
        .setDescriptorSetCount(1)
        .setPSetLayouts(&descriptorSetLayout);

    vk::DescriptorSet descriptorSet = device.allocateDescriptorSets(descriptorSetAllocateInfo)[0];

    // Update the descriptor set
    vk::DescriptorBufferInfo descriptorBufferInfo = vk::DescriptorBufferInfo()
        .setBuffer(buffer)
        .setOffset(0)
        .setRange(VK_WHOLE_SIZE);

    vk::WriteDescriptorSet writeDescriptorSet = vk::WriteDescriptorSet()
        .setDstSet(descriptorSet)
        .setDstBinding(0)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setPBufferInfo(&descriptorBufferInfo);

    device.updateDescriptorSets(1, &writeDescriptorSet, 0, nullptr);

    // Create a compute shader module
    const char* computeShaderCode = R"(
        #version 450

        layout(set = 0, binding = 0) buffer InputBuffer
        {
            float data[];
        } inputBuffer;

        layout(set = 0, binding = 1) buffer OutputBuffer
        {
            float data[];
        } outputBuffer;

        void main()
        {
            uint index = gl_GlobalInvocationID.x;
            outputBuffer.data[index] = inputBuffer.data[index];
        }
    )";

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo = vk::ShaderModuleCreateInfo()
        .setCodeSize(strlen(computeShaderCode))
        .setPCode((uint32_t*)computeShaderCode);

    vk::ShaderModule computeShaderModule = device.createShaderModule(shaderModuleCreateInfo);

    // Create a compute pipeline
    vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(computeShaderModule)
        .setPName("main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo = vk::ComputePipelineCreateInfo()
        .setLayout(pipelineLayout)
        .setStage(pipelineShaderStageCreateInfo);

    auto computePipelineResult = device.createComputePipeline(nullptr, computePipelineCreateInfo);
    if (computePipelineResult.result != vk::Result::eSuccess)
    {
        std::cerr << "Failed to create Vulkan compute pipeline." << std::endl;
        return false;
    }
    vk::Pipeline computePipeline = computePipelineResult.value; 

    // Create a command pool
    vk::CommandPoolCreateInfo commandPoolCreateInfo = vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(computeQueueFamilyIndex);

    vk::CommandPool commandPool = device.createCommandPool(commandPoolCreateInfo);

    // Create a command buffer
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo = vk::CommandBufferAllocateInfo()
        .setCommandPool(commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);

    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(commandBufferAllocateInfo)[0];

    // Begin the command buffer
    vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    commandBuffer.begin(commandBufferBeginInfo);

    // Bind the pipeline
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);

    // Bind the descriptor set
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Dispatch the compute shader
    commandBuffer.dispatch(width * height * channels, 1, 1);

    // End the command buffer
    commandBuffer.end();

    // Submit the command buffer
    vk::SubmitInfo submitInfo = vk::SubmitInfo()
        .setCommandBufferCount(1)
        .setPCommandBuffers(&commandBuffer);

    queue.submit(1, &submitInfo, nullptr);
    queue.waitIdle();

    // Destroy the command buffer
    device.freeCommandBuffers(commandPool, 1, &commandBuffer);

    // Destroy the command pool
    device.destroyCommandPool(commandPool);

    // Destroy the compute pipeline
    device.destroyPipeline(computePipeline);

    // Destroy the compute shader module
    device.destroyShaderModule(computeShaderModule);

    // Destroy the descriptor pool
    device.destroyDescriptorPool(descriptorPool);

    // Destroy the descriptor set layout
    device.destroyDescriptorSetLayout(descriptorSetLayout);

    // Destroy the pipeline layout
    device.destroyPipelineLayout(pipelineLayout);

    // Destroy the buffer
    device.destroyBuffer(buffer);

    // Free the memory allocated for the buffer
    device.freeMemory(memory);

    // Destroy the buffer
    device.destroyBuffer(cubeMapBuffer);

    // Free the memory allocated for the buffer
    device.freeMemory(cubeMapMemory);


    //////////////////////////////////////////////////////////////////////////
    //// Vulkan cleanup code
    //////////////////////////////////////////////////////////////////////////

    // Destroy the Vulkan device and instance
    device.destroy();
    instance.destroy();

    return true;
}

void converter::save(std::string output_file)
{
    // Save the radiance image in EXR format
    SaveEXR(data, width, height, output_file.c_str());    
}