#include <Hasty/Vulkan.h>
#include <Hasty/Image.h>

#include <tiny_obj_loader.h>

using namespace Hasty;

class RaytracerPrototype {
public:
  VulkanInstance instance;
  std::shared_ptr<VulkanComputeDeviceAndQueue> deviceAndQueue;

  PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR{ nullptr };
  PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR{ nullptr };
  PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR{ nullptr };
  PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR{ nullptr };
  PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR{ nullptr };

#define _VK_HASTY_LOAD_FUNCTION(logicalDevice, x)  pfn ## _ ## x = reinterpret_cast<decltype(pfn ## _ ## x)>(vkGetDeviceProcAddr(logicalDevice, #x));

  VkAccelerationStructureKHR accelerationStructureBottomLevel{ nullptr };
  std::unique_ptr<VulkanBuffer> accelerationStructureBottomLevelBuffer;

  VkAccelerationStructureKHR accelerationStructureTopLevel{ nullptr };
  std::unique_ptr<VulkanBuffer> accelerationStructureTopLevelBuffer;

  void destroy() {
    if (deviceAndQueue) {
      deviceAndQueue->waitDeviceIdle();

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

  void buildBottomLevel(
    const std::shared_ptr<VulkanComputeDeviceAndQueue>& deviceAndQueue,
    const std::vector<float>& objVertices,
    const std::vector<uint32_t>& objIndices) {
    std::size_t objVerticesByteCount = sizeof(tinyobj::real_t) * objVertices.size();
    std::size_t objIndicesByteCount = sizeof(uint32_t) * objIndices.size();

    const VkBufferUsageFlags bufferUsageFlags =
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    VulkanBuffer vertexBuffer(deviceAndQueue, objVerticesByteCount, bufferUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VulkanBuffer indexBuffer(deviceAndQueue, objIndicesByteCount, bufferUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vertexBuffer.loadData((void*)objVertices.data(), objVerticesByteCount);
    indexBuffer.loadData((void*)objIndices.data(), objIndicesByteCount);

    deviceAndQueue->waitQueueIdle();

    VkDeviceAddress vertexBufferDeviceAddress = vertexBuffer.getDeviceAddress(deviceAndQueue->getLogicalDevice());
    VkDeviceAddress indexBufferDeviceAddress = indexBuffer.getDeviceAddress(deviceAndQueue->getLogicalDevice());

    VkAccelerationStructureGeometryTrianglesDataKHR geometryTrianglesData{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
    geometryTrianglesData.indexType = VK_INDEX_TYPE_UINT32;
    geometryTrianglesData.indexData.deviceAddress = indexBufferDeviceAddress;
    geometryTrianglesData.maxVertex = static_cast<uint32_t>(objVertices.size() / 3 - 1);
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
    buildRangeInfo.primitiveCount = static_cast<uint32_t>(objIndices.size() / 3);
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
  void buildTopLevel(
    const std::shared_ptr<VulkanComputeDeviceAndQueue>& deviceAndQueue) {

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
    instancesBuffer.loadData(&instance, sizeof(instance));

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

  void buildAccelerationStructure(
    const std::shared_ptr<VulkanComputeDeviceAndQueue>& deviceAndQueue,
    const std::vector<float> &objVertices,
    const std::vector<uint32_t> &objIndices) {

    buildBottomLevel(deviceAndQueue, objVertices, objIndices);
    buildTopLevel(deviceAndQueue);
  }

  void run() {
    instance.init();
    deviceAndQueue = std::make_shared<VulkanComputeDeviceAndQueue>();
    deviceAndQueue->init(instance);

    loadFunctions(deviceAndQueue->getLogicalDevice());

    VkDeviceSize render_width = 1280;
    VkDeviceSize render_height = 720;
    VkDeviceSize bufferSizeBytes = render_width * render_height * 3 * sizeof(float);

    VulkanBuffer buffer(deviceAndQueue, bufferSizeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    tinyobj::ObjReader reader;
    reader.ParseFromFile("./CornellBox.obj");

    const std::vector<tinyobj::real_t> objVertices = reader.GetAttrib().GetVertices();
    static_assert(sizeof(tinyobj::real_t) == 4);
    const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();
    assert(objShapes.size() == 1);
    const tinyobj::shape_t& objShape = objShapes[0];
    std::vector<uint32_t> objIndices;
    objIndices.reserve(objShape.mesh.indices.size());
    for (const tinyobj::index_t& index : objShape.mesh.indices)
    {
      objIndices.push_back(index.vertex_index);
    }

    buildAccelerationStructure(deviceAndQueue, objVertices, objIndices);

    destroy();
  }
};

int main(int argc, const char** argv)
{
  RaytracerPrototype raytracer;
  raytracer.run();

  return 0;
}

