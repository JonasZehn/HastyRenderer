#include <Hasty/Vulkan.h>
#include <Hasty/Image.h>
#include <Hasty/File.h>
#include <Hasty/RenderJob.h>
#include <Hasty/BRDF.h>
#include <Hasty/Scene.h>

#include "VulkanInitializers.hpp"

#include <tiny_obj_loader.h>

using namespace Hasty;

#define _VK_HASTY_LOAD_FUNCTION(logicalDevice, x)  pfn ## _ ## x = reinterpret_cast<decltype(pfn ## _ ## x)>(vkGetDeviceProcAddr(logicalDevice, #x));

struct CameraUniformBufferObject {
  alignas(4) Vec3f position;
  alignas(4) float fovSlope;
  alignas(4) Vec3f forward;
  alignas(4) Vec3f right;
  alignas(4) Vec3f up;
};

class RaytracerPrototype {
public:

  void destroy() {
    if (deviceAndQueue) {
      deviceAndQueue->waitDeviceIdle();
      VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

      if (semaphore) {
        vkDestroySemaphore(logicalDevice, semaphore, nullptr);
        semaphore = nullptr;
      }
      if (pipeline) {
        vkDestroyPipeline(logicalDevice, pipeline, nullptr);
        pipeline = nullptr;
      }
      if (pipelineLayout) {
        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
        pipelineLayout = nullptr;
      }
      if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
        descriptorSetLayout = nullptr;
      }
      if (descriptorPool) {
        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
        descriptorPool = nullptr;
      }
      if (accelerationStructureTopLevel) {
        pfn_vkDestroyAccelerationStructureKHR(deviceAndQueue->getLogicalDevice(), accelerationStructureTopLevel, nullptr);
        accelerationStructureTopLevel = nullptr;
      }
      if (accelerationStructureBottomLevel) {
        pfn_vkDestroyAccelerationStructureKHR(deviceAndQueue->getLogicalDevice(), accelerationStructureBottomLevel, nullptr);
        accelerationStructureBottomLevel = nullptr;
      }
      accelerationStructureBottomLevelBuffer = nullptr;

      deviceAndQueue = nullptr;
    }
  }

  void loadFunctions(VkDevice logicalDevice) {
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkGetAccelerationStructureBuildSizesKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkCreateAccelerationStructureKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkCmdBuildAccelerationStructuresKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkDestroyAccelerationStructureKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkGetAccelerationStructureDeviceAddressKHR);
  }

  void transferMeshesToGPU(Scene &scene) {
    std::vector<float> vertices = scene.getVertices();
    std::vector<std::size_t> geometryIDs = scene.getGeometryIDs();
    vertexCount = vertices.size() / 3;
    faceCount = 0;
    for (std::size_t geometryID : geometryIDs) {
      faceCount += scene.getTriangleCount(geometryID);
    }
    std::vector<uint32_t> indices;
    indices.reserve(faceCount * 3);
    std::vector<float> colors;
    colors.reserve(faceCount * 3);
    std::size_t faceOffset = 0;
    for (std::size_t geometryID : geometryIDs) {
      std::size_t geometryFaceCount = scene.getTriangleCount(geometryID);
      for (std::size_t primIndex = 0; primIndex < geometryFaceCount; primIndex++)
      {
        std::array<int, 3> tri = scene.getTriangleVertexIndices(geometryID, primIndex);
        indices.push_back(tri[0]);
        indices.push_back(tri[1]);
        indices.push_back(tri[2]);

        BXDF& bxdf = scene.getBXDF(geometryID, primIndex);
        PrincipledBRDF* principledBRDF = dynamic_cast<PrincipledBRDF*>(&bxdf);

        Vec3f color(0.5f, 0.5f, 0.5f);
        if (principledBRDF != nullptr) {
          ITextureMap3f &albedoMap = principledBRDF->getAlbedo();
          ConstantTexture3f* albedoConstant = dynamic_cast<ConstantTexture3f*>(&albedoMap);
          if (albedoConstant != nullptr) {
            color = albedoConstant->getValue();
          }
        }

        colors.push_back(color[0]);
        colors.push_back(color[1]);
        colors.push_back(color[2]);
      }
      faceOffset += geometryFaceCount;
    }
    assert(faceOffset == faceCount);

    std::size_t verticesByteCount = sizeof(float) * vertices.size();
    std::size_t indicesByteCount = sizeof(uint32_t) * indices.size();
    std::size_t colorsByteCount = sizeof(float) * colors.size();

    const VkBufferUsageFlags bufferUsageFlags =
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    vertexBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, verticesByteCount, bufferUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vertexBuffer->write((void*)vertices.data(), verticesByteCount);

    indexBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, indicesByteCount, bufferUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    indexBuffer->write((void*)indices.data(), indicesByteCount);

    colorBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, colorsByteCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    colorBuffer->write((void*)colors.data(), colorsByteCount);

    CameraUniformBufferObject cameraData;
    cameraData.position = scene.camera.getPosition();
    cameraData.fovSlope = scene.camera.getFoVSlope();
    cameraData.forward = scene.camera.getForward();
    cameraData.up = scene.camera.getUp();
    cameraData.right = scene.camera.getRight();
    cameraUniformBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, sizeof(cameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    cameraUniformBuffer->write((void*)&cameraData, sizeof(cameraData));

    deviceAndQueue->waitQueueIdle();
  }
  void buildBottomLevel() {

    VkDeviceAddress vertexBufferDeviceAddress = vertexBuffer->getDeviceAddress(deviceAndQueue->getLogicalDevice());
    VkDeviceAddress indexBufferDeviceAddress = indexBuffer->getDeviceAddress(deviceAndQueue->getLogicalDevice());

    VkAccelerationStructureGeometryTrianglesDataKHR geometryTrianglesData{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
    geometryTrianglesData.indexType = VK_INDEX_TYPE_UINT32;
    geometryTrianglesData.indexData.deviceAddress = indexBufferDeviceAddress;
    geometryTrianglesData.maxVertex = static_cast<uint32_t>(vertexCount - 1);
    geometryTrianglesData.transformData.deviceAddress = 0;
    geometryTrianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometryTrianglesData.vertexData.deviceAddress = vertexBufferDeviceAddress;
    geometryTrianglesData.vertexStride = 3 * sizeof(float);

    VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = geometryTrianglesData;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo;
    buildRangeInfo.firstVertex = 0;
    buildRangeInfo.primitiveCount = static_cast<uint32_t>(faceCount);
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.transformOffset = 0;

    VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();
    VkCommandBuffer cmdBuf = deviceAndQueue->beginSingleTimeCommandBuffer();
    VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };

    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildGeometryInfo.geometryCount = 1;
    buildGeometryInfo.pGeometries = &geometry;
    buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    uint32_t maxPrimitiveCounts = buildRangeInfo.primitiveCount;

    pfn_vkGetAccelerationStructureBuildSizesKHR(
      logicalDevice,
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildGeometryInfo,
      &maxPrimitiveCounts,
      &buildSizesInfo);
    std::size_t buildScratchSize = buildSizesInfo.buildScratchSize;

    VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    createInfo.size = buildSizesInfo.accelerationStructureSize;

    VulkanBuffer scratchBuffer(deviceAndQueue, buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceAddress scratchAddress = scratchBuffer.getDeviceAddress(logicalDevice);

    accelerationStructureBottomLevelBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, createInfo.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
      | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createInfo.buffer = accelerationStructureBottomLevelBuffer->getRaw();
    VK_CHECK_RESULT(pfn_vkCreateAccelerationStructureKHR(logicalDevice, &createInfo, nullptr, &accelerationStructureBottomLevel));

    buildGeometryInfo.dstAccelerationStructure = accelerationStructureBottomLevel;
    buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

    const VkAccelerationStructureBuildRangeInfoKHR* buildRangeInfoPtr = &buildRangeInfo;
    pfn_vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildGeometryInfo, &buildRangeInfoPtr);

    VulkanFence fence(deviceAndQueue);
    deviceAndQueue->endSingleTimeCommandBuffer(cmdBuf, fence.getRaw());
    deviceAndQueue->waitForFence(fence);
  }
  void buildTopLevel() {

    VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();
    VkCommandBuffer cmdBuf = deviceAndQueue->beginSingleTimeCommandBuffer();

    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
    deviceAddressInfo.accelerationStructure = accelerationStructureBottomLevel;
    VkDeviceAddress bottomLevelDeviceAddress = pfn_vkGetAccelerationStructureDeviceAddressKHR(logicalDevice, &deviceAddressInfo);

    VkAccelerationStructureInstanceKHR instance{};
    instance.accelerationStructureReference = bottomLevelDeviceAddress;
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.mask = 0xFF;

    VulkanBuffer instancesBuffer(deviceAndQueue, sizeof(instance), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    instancesBuffer.write(&instance, sizeof(instance));

    uint32_t maxPrimitiveCounts = 1;
    VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
    geometryInstances.arrayOfPointers = VK_FALSE;
    geometryInstances.data.deviceAddress = instancesBuffer.getDeviceAddress(logicalDevice);
    
    VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = geometryInstances;

    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildGeometryInfo.geometryCount = 1;
    buildGeometryInfo.pGeometries = &geometry;
    buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    pfn_vkGetAccelerationStructureBuildSizesKHR(
      logicalDevice,
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildGeometryInfo,
      &maxPrimitiveCounts,
      &sizeInfo);

    VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    createInfo.size = sizeInfo.accelerationStructureSize;
    accelerationStructureTopLevelBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, createInfo.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createInfo.buffer = accelerationStructureTopLevelBuffer->getRaw();
    VK_CHECK_RESULT(pfn_vkCreateAccelerationStructureKHR(logicalDevice, &createInfo, nullptr, &accelerationStructureTopLevel));

    VulkanBuffer scratchBuffer(deviceAndQueue, sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceAddress scratchAddress = scratchBuffer.getDeviceAddress(logicalDevice);

    buildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildGeometryInfo.dstAccelerationStructure = accelerationStructureTopLevel;
    buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{ };
    buildRangeInfo.primitiveCount = maxPrimitiveCounts;
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.firstVertex = 0;
    buildRangeInfo.transformOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR* buildRangeInfoPointer = &buildRangeInfo;

    pfn_vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildGeometryInfo, &buildRangeInfoPointer);

    VulkanFence fence(deviceAndQueue);
    deviceAndQueue->endSingleTimeCommandBuffer(cmdBuf, fence.getRaw());
    deviceAndQueue->waitForFence(fence);
  }

  void buildAccelerationStructure() {

    buildBottomLevel();
    buildTopLevel();
  }

  void buildComputeCommandBuffer(uint32_t width, uint32_t height)
  {
    VkQueue queue = deviceAndQueue->getQueue();

    // Flush the queue if we're rebuilding the command buffer after a pipeline change to ensure it's not currently in use
    VK_CHECK_RESULT(vkQueueWaitIdle(queue));

    VkCommandBufferBeginInfo commandBufferBeginInfo = vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

    uint32_t layoutSizeX = 4;
    uint32_t layoutSizeY = 4;
    uint32_t groupCountX = (width + layoutSizeX - 1) / layoutSizeX;
    uint32_t groupCountY = (height + layoutSizeY - 1) / layoutSizeY;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }

  void preprareCompute(const std::shared_ptr<VulkanComputeDeviceAndQueue>& deviceAndQueue, VulkanImage& dstImage) {
    VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

    std::vector<VkDescriptorPoolSize> poolSizes = {
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1),
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
    };
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 1);
    VK_CHECK_RESULT(vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool));

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT, 0),
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1),
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logicalDevice, &descriptorLayout, nullptr, &descriptorSetLayout));

    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

    VK_CHECK_RESULT(vkAllocateDescriptorSets(logicalDevice, &allocInfo, &descriptorSet));

    VkWriteDescriptorSetAccelerationStructureKHR writeDescriptorSetAccelerationStructureKHR{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    writeDescriptorSetAccelerationStructureKHR.accelerationStructureCount = 1;
    writeDescriptorSetAccelerationStructureKHR.pAccelerationStructures = &accelerationStructureTopLevel;

    VkDescriptorBufferInfo vertexDescriptorBufferInfo = vertexBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo indexDescriptorBufferInfo = indexBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo colorDescriptorBufferInfo = colorBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo cameraDescriptorBufferInfo = cameraUniformBuffer->descriptorBufferInfo();

    std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
      writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0, &writeDescriptorSetAccelerationStructureKHR),
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &dstImage.getDescriptor()),
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &vertexDescriptorBufferInfo),
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &indexDescriptorBufferInfo),
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &colorDescriptorBufferInfo),
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5, &cameraDescriptorBufferInfo)
    };
    vkUpdateDescriptorSets(logicalDevice, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);

    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

    VK_CHECK_RESULT(vkCreatePipelineLayout(logicalDevice, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(pipelineLayout, 0);

    std::vector<char> shaderBinary = Hasty::readFile(HASTY_SHADER_PATH / "rayQuery.comp.spv");
    shader.init(deviceAndQueue, shaderBinary);

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = shader.getModule();
    computeShaderStageInfo.pName = "main";

    computePipelineCreateInfo.stage = computeShaderStageInfo;

    VkPipelineCache pipelineCache = nullptr;
    VK_CHECK_RESULT(vkCreateComputePipelines(logicalDevice, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));

    // Create a command buffer for compute operations
    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
      vks::initializers::commandBufferAllocateInfo(
        deviceAndQueue->getCommandPool(),
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1);

    VK_CHECK_RESULT(vkAllocateCommandBuffers(logicalDevice, &cmdBufAllocateInfo, &commandBuffer));

    // Semaphore for compute & graphics sync
    VkSemaphoreCreateInfo semaphoreCreateInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VK_CHECK_RESULT(vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr, &semaphore));

    // Build a single command buffer containing the compute dispatch commands
    buildComputeCommandBuffer(dstImage.getWidth(), dstImage.getHeight());
  }
  void run(RenderJob &job) {
    VkDeviceSize render_width = job.renderSettings.width;
    VkDeviceSize render_height = job.renderSettings.height;

    instance.init();
    deviceAndQueue = std::make_shared<VulkanComputeDeviceAndQueue>();
    deviceAndQueue->init(instance);

    loadFunctions(deviceAndQueue->getLogicalDevice());

    Image4f image(render_width, render_height);
    VkDeviceSize bufferSizeBytes = render_width * render_height * 4 * sizeof(float);

    VulkanBuffer buffer(deviceAndQueue, bufferSizeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    transferMeshesToGPU(*job.scene);
    buildAccelerationStructure();

    VulkanImage gpuDstImage = allocateDeviceImage(deviceAndQueue, VK_FORMAT_R32G32B32A32_SFLOAT, image.getWidth(), image.getHeight(), VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    deviceAndQueue->transitionImageLayout(gpuDstImage, VK_IMAGE_LAYOUT_GENERAL);
    preprareCompute(deviceAndQueue, gpuDstImage);

    // Wait for rendering finished
    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &commandBuffer;
    computeSubmitInfo.waitSemaphoreCount = 0;
    computeSubmitInfo.pWaitSemaphores = VK_NULL_HANDLE;
    computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores = &semaphore;
    VK_CHECK_RESULT(vkQueueSubmit(deviceAndQueue->getQueue(), 1, &computeSubmitInfo, VK_NULL_HANDLE));

    deviceAndQueue->waitQueueIdle();

    deviceAndQueue->transitionImageLayout(gpuDstImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VulkanFence fence(deviceAndQueue);
    deviceAndQueue->copyImageToBuffer(gpuDstImage, buffer, fence);
    deviceAndQueue->waitForFence(fence);

    buffer.read(image.data(), bufferSizeBytes);

    Hasty::writeEXR(image, "output.exr");

    destroy();
  }

private:
  VulkanInstance instance;
  std::shared_ptr<VulkanComputeDeviceAndQueue> deviceAndQueue;

  std::unique_ptr<VulkanBuffer> vertexBuffer;
  std::size_t vertexCount;
  std::size_t faceCount;
  std::unique_ptr<VulkanBuffer> indexBuffer;
  std::unique_ptr<VulkanBuffer> colorBuffer;

  std::unique_ptr<VulkanBuffer> cameraUniformBuffer;

  PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR{ nullptr };
  PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR{ nullptr };
  PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR{ nullptr };
  PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR{ nullptr };
  PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR{ nullptr };

  VkAccelerationStructureKHR accelerationStructureBottomLevel{ nullptr };
  std::unique_ptr<VulkanBuffer> accelerationStructureBottomLevelBuffer;

  VkAccelerationStructureKHR accelerationStructureTopLevel{ nullptr };
  std::unique_ptr<VulkanBuffer> accelerationStructureTopLevelBuffer;

  VulkanShaderModule shader;
  VkPipeline pipeline{ nullptr };
  VkPipelineLayout pipelineLayout{ nullptr };
  VkSemaphore semaphore{ nullptr };
  VkCommandBuffer commandBuffer{ nullptr };
  VkDescriptorPool descriptorPool{ nullptr };
  VkDescriptorSetLayout descriptorSetLayout{ nullptr };
  VkDescriptorSet descriptorSet{ nullptr };
};

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cout << " unexpected number of arguments: " << argc << " expecting 3, usage <binary> <renderJobFile> <outputDirectory> " << std::endl;
    return 1;
  }

  std::filesystem::path outputFolder = argv[2];
  if (!std::filesystem::is_directory(outputFolder))
  {
    std::cout << "error given output path " << outputFolder << " is not a directory " << std::endl;
    return 1;
  }

  RenderJob job;
  try
  {
    job.loadJSON(argv[1]);
  }
  catch (const std::exception& e)
  {
    std::cout << "exception: " << e.what() << std::endl;
    return 2;
  }

  RaytracerPrototype raytracer;
  raytracer.run(job);

  return 0;
}

