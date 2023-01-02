#include <Hasty/Vulkan.h>
#include <Hasty/Image.h>
#include <Hasty/File.h>
#include <Hasty/RenderJob.h>
#include <Hasty/BRDF.h>
#include <Hasty/Scene.h>

#include "VulkanInitializers.hpp"
#include <ViewerUtils.h>

#include <tiny_obj_loader.h>

#include <variant>

using namespace Hasty;

#define _VK_HASTY_LOAD_FUNCTION(logicalDevice, x)  pfn ## _ ## x = reinterpret_cast<decltype(pfn ## _ ## x)>(vkGetDeviceProcAddr(logicalDevice, #x));

struct CameraUniformBufferObject
{
  alignas(4) Vec3f position;
  alignas(4) float fovSlope;
  alignas(4) Vec3f forward;
  alignas(4) Vec3f right;
  alignas(4) Vec3f up;
  alignas(4) float apertureSize;
  alignas(4) float focalDistance;
  alignas(4) int32_t numBlades;
  alignas(4) float bladeRotation;
};
struct MaterialUniformBufferObject
{
  alignas(4) Vec3f emission;
  alignas(4) float specular;
  alignas(4) float anisotropy;
  alignas(4) float transmission;
  alignas(4) float indexOfRefraction;
};
struct PushConstantsSample
{
  alignas(4) int32_t nSamples;
  alignas(4) Vec3f backgroundColor;
};


template<typename _PixelType>
struct VulkanPixelTraits
{

};
template<>
struct VulkanPixelTraits < float >
{
  static const VkFormat Format = VK_FORMAT_R32_SFLOAT;
};
template<>
struct VulkanPixelTraits<Vec4f>
{
  static const VkFormat Format = VK_FORMAT_R32G32B32A32_SFLOAT;
};

class RaytracerPrototype
{
public:

  void destroy()
  {
    if(deviceAndQueue)
    {
      deviceAndQueue->waitDeviceIdle();
      VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

      if(pipeline)
      {
        vkDestroyPipeline(logicalDevice, pipeline, nullptr);
        pipeline = nullptr;
      }
      if(pipelineLayout)
      {
        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
        pipelineLayout = nullptr;
      }
      if(descriptorSetLayout)
      {
        vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
        descriptorSetLayout = nullptr;
      }
      if(descriptorPool)
      {
        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
        descriptorPool = nullptr;
      }
      if(accelerationStructureTopLevel)
      {
        pfn_vkDestroyAccelerationStructureKHR(deviceAndQueue->getLogicalDevice(), accelerationStructureTopLevel, nullptr);
        accelerationStructureTopLevel = nullptr;
      }
      if(accelerationStructureBottomLevel)
      {
        pfn_vkDestroyAccelerationStructureKHR(deviceAndQueue->getLogicalDevice(), accelerationStructureBottomLevel, nullptr);
        accelerationStructureBottomLevel = nullptr;
      }
      accelerationStructureBottomLevelBuffer = nullptr;

      deviceAndQueue = nullptr;
    }
  }

  void loadFunctions(VkDevice logicalDevice)
  {
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkGetAccelerationStructureBuildSizesKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkCreateAccelerationStructureKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkCmdBuildAccelerationStructuresKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkDestroyAccelerationStructureKHR);
    _VK_HASTY_LOAD_FUNCTION(logicalDevice, vkGetAccelerationStructureDeviceAddressKHR);
  }

  template<typename _PixelType>
  void allocateAndTransferImageToGPUAndTransitionLayout(const Image<_PixelType>& imageCPU, std::unique_ptr<VulkanBuffer>& imageBuffer, std::unique_ptr<VulkanImage>& imageGPU, VkImageLayout finalImageLayout)
  {
    std::size_t imageByteCount = sizeof(_PixelType) * imageCPU.size();
    imageBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, imageByteCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    imageBuffer->write((void*)imageCPU.data(), imageByteCount);

    imageGPU = std::make_unique<VulkanImage>(
      deviceAndQueue,
      imageCPU.getWidth(),
      imageCPU.getHeight(),
      VulkanPixelTraits<_PixelType>::Format,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    deviceAndQueue->transitionImageLayout(*imageGPU, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    deviceAndQueue->copyBufferToImage(*imageBuffer, *imageGPU);

    deviceAndQueue->transitionImageLayout(*imageGPU, finalImageLayout);
  }

  // value type for a constant value or a pointer to a large existing image
  template<typename _PixelType>
  struct ImagePtrOrConstant
  {
    typedef Image<_PixelType> ValueType;
    typedef Image<_PixelType> const* PtrType;

    ImagePtrOrConstant(ValueType val) :value(val)
    {
    }
    ImagePtrOrConstant(PtrType ptr) :value(ptr)
    {

    }

    Image<_PixelType> const* getPtr() const
    {
      if(std::holds_alternative<ValueType>(value))
      {
        return &std::get<ValueType>(value);
      }
      else
      {
        return std::get<PtrType>(value);
      }
    }

  private:
    std::variant<ValueType, PtrType> value;
  };

  template<typename _PixelType>
  ImagePtrOrConstant<_PixelType> getImagePointer(ITextureMap<_PixelType> const* textureMap)
  {
    ConstantTexture<_PixelType> const* constantTexture = dynamic_cast<ConstantTexture<_PixelType> const*>(textureMap);
    Texture<_PixelType> const* texture = dynamic_cast<Texture<_PixelType> const*>(textureMap);

    if(constantTexture != nullptr)
    {
      Image<_PixelType> constantImage(1, 1);
      constantImage(0, 0) = constantTexture->getValue();
      return constantImage;
    }
    else if(texture != nullptr)
    {
      return &texture->getImage();
    }
    else
    {
      throw std::runtime_error("unsupported/not implemented input texture");
    }
    return nullptr;
  }

  void transferDataToGPU(Scene& scene, uint32_t renderWidth, uint32_t renderHeight, Vec3f& backgroundColor)
  {
    std::vector<float> vertices = scene.getVertices();
    std::vector<std::size_t> geometryIDs = scene.getGeometryIDs();
    vertexCount = vertices.size() / 3;
    faceCount = 0;
    for(std::size_t geometryID : geometryIDs)
    {
      faceCount += scene.getTriangleCount(geometryID);
    }
    std::vector<Vec2f> textureCoordinates;
    textureCoordinates.reserve(faceCount * 3);
    std::vector<uint32_t> indices;
    indices.reserve(faceCount * 3);
    std::vector<uint32_t> materialIndices;
    materialIndices.reserve(faceCount);
    std::vector<MaterialUniformBufferObject> materials;
    materials.reserve(scene.getMaterialCount());

    backgroundColor = Vec3f::Zero();

    BackgroundColor const * backgroundColorObject = dynamic_cast<BackgroundColor const *>(&scene.getBackground());
    if(backgroundColorObject != nullptr)
    {
      backgroundColor = backgroundColorObject->getColor();
    }

    for(int materialIdx = 0; materialIdx < scene.getMaterialCount(); materialIdx++)
    {
      BXDF& bxdf = scene.getBXDFByIndex(materialIdx);
      PrincipledBRDF* principledBRDF = dynamic_cast<PrincipledBRDF*>(&bxdf);

      MaterialUniformBufferObject material;
      material.emission = scene.getMaterialEmission(materialIdx);
      material.specular = 0.0f;
      material.transmission = 0.0f;
      material.transmission = 0.0f;
      material.indexOfRefraction = 1.0f;

      Image1f constImage1f(1, 1);
      constImage1f(0, 0) = 0.0f;
      Image3f constImage3f(1, 1);
      constImage3f(0, 0) = Vec3f::Zero();

      ImagePtrOrConstant<Vec3f> albedoImage3(constImage3f);
      ImagePtrOrConstant<float> metallicImage(constImage1f);
      ImagePtrOrConstant<float> roughnessImage(constImage1f);
      if(material.emission == Vec3f::Zero())
      {
        if(principledBRDF != nullptr)
        {
          albedoImage3 = getImagePointer<Vec3f>(&principledBRDF->getAlbedo());
          metallicImage = getImagePointer<float>(&principledBRDF->getMetallic());
          roughnessImage = getImagePointer<float>(&principledBRDF->getRoughness());

          material.specular = principledBRDF->getSpecular();
          material.anisotropy = principledBRDF->getAnisotropy();
          material.transmission = principledBRDF->getTransmission();
          material.indexOfRefraction = principledBRDF->getIndexOfRefractionMap();
        }
      }

      materials.push_back(material);


      Image4f albedoImage4 = addAlphaChannel(*(albedoImage3.getPtr()));

      albedoImageBuffers.emplace_back();
      albedoImages.emplace_back();
      allocateAndTransferImageToGPUAndTransitionLayout<Vec4f>(albedoImage4, albedoImageBuffers.back(), albedoImages.back(), VK_IMAGE_LAYOUT_GENERAL);

      metallicImageBuffers.emplace_back();
      metallicImages.emplace_back();
      allocateAndTransferImageToGPUAndTransitionLayout<float>(*metallicImage.getPtr(), metallicImageBuffers.back(), metallicImages.back(), VK_IMAGE_LAYOUT_GENERAL);

      roughnessImageBuffers.emplace_back();
      roughnessImages.emplace_back();
      allocateAndTransferImageToGPUAndTransitionLayout<float>(*roughnessImage.getPtr(), roughnessImageBuffers.back(), roughnessImages.back(), VK_IMAGE_LAYOUT_GENERAL);

    }

    std::size_t faceOffset = 0;
    for(std::size_t geometryID : geometryIDs)
    {
      std::size_t geometryFaceCount = scene.getTriangleCount(geometryID);
      for(std::size_t primIndex = 0; primIndex < geometryFaceCount; primIndex++)
      {
        std::array<int, 3> tri = scene.getTriangleVertexIndices(geometryID, primIndex);
        indices.push_back(tri[0]);
        indices.push_back(tri[1]);
        indices.push_back(tri[2]);

        materialIndices.push_back(scene.getMaterialIndex(geometryID, primIndex));

        std::array<Vec2f, 3> uv = { Vec2f::Zero(), Vec2f::Zero(), Vec2f::Zero() };
        std::optional< std::array<Vec2f, 3> > uvOptional = scene.getTriangleUV(geometryID, primIndex);
        if(uvOptional.has_value()) uv = uvOptional.value();

        textureCoordinates.push_back(uv[0]);
        textureCoordinates.push_back(uv[1]);
        textureCoordinates.push_back(uv[2]);
      }
      faceOffset += geometryFaceCount;
    }
    assert(faceOffset == faceCount);

    uint64_t seed = 0;
    RNG rng(seed);
    std::vector<uint32_t> randomInputState;
    int numUintsPerPixel = 3;
    randomInputState.reserve(renderWidth * renderHeight * numUintsPerPixel);
    for(int i = 0; i < renderWidth; i++)
    {
      for(int j = 0; j < renderHeight; j++)
      {
        for(int k = 0; k < numUintsPerPixel; k++)
        {
          randomInputState.push_back(rng());
        }
      }
    }

    std::size_t verticesByteCount = sizeof(decltype(vertices)::value_type) * vertices.size();
    std::size_t textureCoordinatesByteCount = sizeof(decltype(textureCoordinates)::value_type) * textureCoordinates.size();
    std::size_t indicesByteCount = sizeof(decltype(indices)::value_type) * indices.size();
    std::size_t materialIndicesByteCount = sizeof(decltype(materialIndices)::value_type) * materialIndices.size();
    std::size_t materialsByteCount = sizeof(decltype(materials)::value_type) * materials.size();
    std::size_t randomInputStateByteCount = sizeof(decltype(randomInputState)::value_type) * randomInputState.size();

    const VkBufferUsageFlags bufferUsageFlags =
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags accelerationStructureBufferUsageFlags =
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    vertexBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, verticesByteCount, accelerationStructureBufferUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vertexBuffer->write((void*)vertices.data(), verticesByteCount);

    textureCoordinatesBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, textureCoordinatesByteCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    textureCoordinatesBuffer->write((void*)textureCoordinates.data(), textureCoordinatesByteCount);

    indexBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, indicesByteCount, accelerationStructureBufferUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    indexBuffer->write((void*)indices.data(), indicesByteCount);

    materialIndicesBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, materialIndicesByteCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    materialIndicesBuffer->write((void*)materialIndices.data(), materialIndicesByteCount);

    materialBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, materialsByteCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    materialBuffer->write((void*)materials.data(), materialsByteCount);

    randomInputStateBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, randomInputStateByteCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    randomInputStateBuffer->write((void*)randomInputState.data(), randomInputStateByteCount);

    CameraUniformBufferObject cameraData;
    cameraData.position = scene.camera.getPosition();
    cameraData.fovSlope = scene.camera.getFoVSlope();
    cameraData.forward = scene.camera.getForward();
    cameraData.up = scene.camera.getUp();
    cameraData.right = scene.camera.getRight();
    cameraData.apertureSize = scene.camera.getApertureSize();
    cameraData.focalDistance = scene.camera.getFocalDistance();
    cameraData.numBlades = scene.camera.getNumBlades();
    cameraData.bladeRotation = scene.camera.getBladeRotation();

    cameraUniformBuffer = std::make_unique<VulkanBuffer>(deviceAndQueue, sizeof(cameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    cameraUniformBuffer->write((void*)&cameraData, sizeof(cameraData));

    deviceAndQueue->waitQueueIdle();
  }
  void buildBottomLevel()
  {

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
  void buildTopLevel()
  {

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

  void buildAccelerationStructure()
  {

    buildBottomLevel();
    buildTopLevel();
  }

  void buildComputeCommandBuffer(VkCommandBuffer commandBuffer, int32_t numSamplesPerRun, const Vec3f& backgroundColor, uint32_t width, uint32_t height)
  {
    VkCommandBufferBeginInfo commandBufferBeginInfo = vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
    PushConstantsSample pushConstants;
    pushConstants.nSamples = numSamplesPerRun;
    pushConstants.backgroundColor = backgroundColor;
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSample), &pushConstants);

    uint32_t layoutSizeX = 4;
    uint32_t layoutSizeY = 4;
    uint32_t groupCountX = (width + layoutSizeX - 1) / layoutSizeX;
    uint32_t groupCountY = (height + layoutSizeY - 1) / layoutSizeY;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }

  void preprareCompute(const std::shared_ptr<VulkanComputeDeviceAndQueue>& deviceAndQueue, int numSamplesPerRun, VulkanImage& colorImage, VulkanImage& normalImage, VulkanImage& albedoImage)
  {
    VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

    std::vector<VkDescriptorPoolSize> poolSizes = {
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1),  // tlas
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1), // Vertices
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1), // TextureCoordinates
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1), // Indices
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1), // MaterialIndices
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1), // Materials
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, albedoImages.size()), // albedoTextures
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, metallicImages.size()), // metallicTextures
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, roughnessImages.size()), // roughnessTextures
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1), // RandomInputState
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1), // Camera Input
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1), // colorImage
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1), // normalImage
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1), // albedoImage
    };
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 1);
    VK_CHECK_RESULT(vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool));

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT, 0), // tlas
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),  // Vertices
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),  // TextureCoordinates
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3), // Indices
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4), // MaterialIndices
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5), // Materials
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 6, albedoImages.size()), // albedoTextures
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 7, metallicImages.size()), // metallicTextures
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 8, roughnessImages.size()), // roughnessTextures
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 9), // RandomInputState
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 10), // Camera Input
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 11), // colorImage
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 12), // normalImage
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 13), // albedoImage
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logicalDevice, &descriptorLayout, nullptr, &descriptorSetLayout));

    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

    VK_CHECK_RESULT(vkAllocateDescriptorSets(logicalDevice, &allocInfo, &descriptorSet));

    VkWriteDescriptorSetAccelerationStructureKHR writeDescriptorSetAccelerationStructureKHR{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    writeDescriptorSetAccelerationStructureKHR.accelerationStructureCount = 1;
    writeDescriptorSetAccelerationStructureKHR.pAccelerationStructures = &accelerationStructureTopLevel;

    VkDescriptorBufferInfo vertexDescriptorBufferInfo = vertexBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo textureCoordinatesDescriptorBufferInfo = textureCoordinatesBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo indexDescriptorBufferInfo = indexBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo materialIndicesDescriptorBufferInfo = materialIndicesBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo materialDescriptorBufferInfo = materialBuffer->descriptorBufferInfo();

    std::vector<VkDescriptorImageInfo> albedoImagesDescriptorBufferInfos(albedoImages.size());
    for(size_t i = 0; i < albedoImages.size(); i++)
    {
      albedoImagesDescriptorBufferInfos[i] = albedoImages[i]->getDescriptor();
    }

    std::vector<VkDescriptorImageInfo> metallicImagesDescriptorBufferInfos(metallicImages.size());
    for(size_t i = 0; i < metallicImages.size(); i++)
    {
      metallicImagesDescriptorBufferInfos[i] = metallicImages[i]->getDescriptor();
    }

    std::vector<VkDescriptorImageInfo> roughnessImagesDescriptorBufferInfos(roughnessImages.size());
    for(size_t i = 0; i < roughnessImages.size(); i++)
    {
      roughnessImagesDescriptorBufferInfos[i] = roughnessImages[i]->getDescriptor();
    }

    VkDescriptorBufferInfo randomInputStateDescriptorBufferInfo = randomInputStateBuffer->descriptorBufferInfo();
    VkDescriptorBufferInfo cameraDescriptorBufferInfo = cameraUniformBuffer->descriptorBufferInfo();

    std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
      writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0, &writeDescriptorSetAccelerationStructureKHR), // tlas
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &vertexDescriptorBufferInfo),  // Vertices
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &textureCoordinatesDescriptorBufferInfo),  // TextureCoordinates
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &indexDescriptorBufferInfo),  // Indices
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &materialIndicesDescriptorBufferInfo), // MaterialIndices
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &materialDescriptorBufferInfo), // Materials
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6, albedoImagesDescriptorBufferInfos.data(), albedoImagesDescriptorBufferInfos.size()), // albedoTextures
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7, metallicImagesDescriptorBufferInfos.data(), metallicImages.size()), // metallicTextures
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8, roughnessImagesDescriptorBufferInfos.data(), roughnessImages.size()), // roughnessTextures
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9, &randomInputStateDescriptorBufferInfo), // RandomInputState
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10, &cameraDescriptorBufferInfo), // Camera Input
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 11, &colorImage.getDescriptor()), // colorImage
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 12, &normalImage.getDescriptor()), // normalImage
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 13, &albedoImage.getDescriptor()), // albedoImage
    };
    vkUpdateDescriptorSets(logicalDevice, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, nullptr);

    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

    std::vector<VkPushConstantRange> pushConstantRanges = {
      vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(PushConstantsSample), 0)
    };

    pPipelineLayoutCreateInfo.pushConstantRangeCount = pushConstantRanges.size();
    pPipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRanges.data();

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


  }
  void run(RenderJob& job, Image3f& colorImage, Image3f& normalImage, Image3f& albedoImage)
  {
    VkDeviceSize renderWidth = job.renderSettings.width;
    VkDeviceSize renderHeight = job.renderSettings.height;

    instance.init();
    deviceAndQueue = std::make_shared<VulkanComputeDeviceAndQueue>();
    deviceAndQueue->init(instance);

    loadFunctions(deviceAndQueue->getLogicalDevice());

    Image4f outputImage;
    outputImage.setZero(renderWidth, renderHeight);

    int nSamplesPerRun = 50;
    int nSamples = job.renderSettings.numSamples;
    int nRuns = (nSamples + nSamplesPerRun - 1) / nSamplesPerRun; // ceil integer division

    VkDeviceSize bufferSizeBytes = outputImage.byteCount();

    VulkanBuffer outputImageBuffer(deviceAndQueue, bufferSizeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    Vec3f backgroundColor;
    transferDataToGPU(*job.scene, renderWidth, renderHeight, backgroundColor);
    buildAccelerationStructure();

    VkClearColorValue clearColorValue;
    for(int i = 0; i < 4; i++) clearColorValue.float32[i] = 0.0;

    VulkanImage colorImageGPU = allocateDeviceImage(deviceAndQueue, VK_FORMAT_R32G32B32A32_SFLOAT, renderWidth, renderHeight, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    deviceAndQueue->transitionImageLayoutAndClear(colorImageGPU, VK_IMAGE_LAYOUT_GENERAL, clearColorValue);

    VulkanImage normalImageGPU = allocateDeviceImage(deviceAndQueue, VK_FORMAT_R32G32B32A32_SFLOAT, renderWidth, renderHeight, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    deviceAndQueue->transitionImageLayoutAndClear(normalImageGPU, VK_IMAGE_LAYOUT_GENERAL, clearColorValue);

    VulkanImage albedoImageGPU = allocateDeviceImage(deviceAndQueue, VK_FORMAT_R32G32B32A32_SFLOAT, renderWidth, renderHeight, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    deviceAndQueue->transitionImageLayoutAndClear(albedoImageGPU, VK_IMAGE_LAYOUT_GENERAL, clearColorValue);

    preprareCompute(deviceAndQueue, nSamplesPerRun, colorImageGPU, normalImageGPU, albedoImageGPU);

    VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

    VkSemaphore lastSemaphore{ nullptr };

    std::vector<VkSemaphore> semaphores;

    int sumSamples = 0;
    for(int i = 0; i < nRuns; i++)
    {
      int nSamplesThisRun = i == nRuns - 1 ? nSamples - nSamplesPerRun * (nRuns - 1) : nSamplesPerRun;
      sumSamples += nSamplesThisRun;

      VkCommandBuffer commandBuffer{ nullptr };
      // Create a command buffer for compute operations
      VkCommandBufferAllocateInfo cmdBufAllocateInfo =
        vks::initializers::commandBufferAllocateInfo(
          deviceAndQueue->getCommandPool(),
          VK_COMMAND_BUFFER_LEVEL_PRIMARY,
          1);

      VK_CHECK_RESULT(vkAllocateCommandBuffers(logicalDevice, &cmdBufAllocateInfo, &commandBuffer));

      VkSemaphore newSemaphore{ nullptr };
      VkSemaphoreCreateInfo semaphoreCreateInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
      VK_CHECK_RESULT(vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr, &newSemaphore));

      buildComputeCommandBuffer(commandBuffer, nSamplesThisRun, backgroundColor, colorImageGPU.getWidth(), colorImageGPU.getHeight());

      VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

      VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
      computeSubmitInfo.commandBufferCount = 1;
      computeSubmitInfo.pCommandBuffers = &commandBuffer;
      if(lastSemaphore == nullptr)
      {
        computeSubmitInfo.waitSemaphoreCount = 0;
        computeSubmitInfo.pWaitSemaphores = VK_NULL_HANDLE;
      }
      else
      {
        computeSubmitInfo.waitSemaphoreCount = 1;
        computeSubmitInfo.pWaitSemaphores = &lastSemaphore;
      }
      computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
      computeSubmitInfo.signalSemaphoreCount = 1;
      computeSubmitInfo.pSignalSemaphores = &newSemaphore;

      std::cout << " submitting " << i << '\n';
      VK_CHECK_RESULT(vkQueueSubmit(deviceAndQueue->getQueue(), 1, &computeSubmitInfo, VK_NULL_HANDLE));

      lastSemaphore = newSemaphore;

      semaphores.push_back(newSemaphore);
    }
    assert(sumSamples == nSamples);

    deviceAndQueue->transitionImageLayout(colorImageGPU, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    deviceAndQueue->transitionImageLayout(normalImageGPU, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    deviceAndQueue->transitionImageLayout(albedoImageGPU, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    {
      VulkanFence fence(deviceAndQueue);
      deviceAndQueue->copyImageToBuffer(colorImageGPU, outputImageBuffer, fence);
      deviceAndQueue->waitForFence(fence);
      outputImageBuffer.read(outputImage.data(), outputImage.byteCount());
      colorImage = removeAlphaChannel(outputImage);
      colorImage /= static_cast<float>(nSamples);
      colorImage *= std::pow(2.0f, job.scene->camera.getExposure());
    }
    {
      VulkanFence fence(deviceAndQueue);
      deviceAndQueue->copyImageToBuffer(normalImageGPU, outputImageBuffer, fence);
      deviceAndQueue->waitForFence(fence);
      outputImageBuffer.read(outputImage.data(), outputImage.byteCount());
      normalImage = removeAlphaChannel(outputImage);
      normalImage /= static_cast<float>(nSamples);
    }
    {
      VulkanFence fence(deviceAndQueue);
      deviceAndQueue->copyImageToBuffer(albedoImageGPU, outputImageBuffer, fence);
      deviceAndQueue->waitForFence(fence);
      outputImageBuffer.read(outputImage.data(), outputImage.byteCount());
      albedoImage = removeAlphaChannel(outputImage);
      albedoImage /= static_cast<float>(nSamples);
    }

    for(auto& semaphore : semaphores)
    {
      vkDestroySemaphore(logicalDevice, semaphore, nullptr);
    }

    destroy();
  }

private:
  VulkanInstance instance;
  std::shared_ptr<VulkanComputeDeviceAndQueue> deviceAndQueue;

  std::unique_ptr<VulkanBuffer> vertexBuffer;
  std::size_t vertexCount;
  std::size_t faceCount;
  std::unique_ptr<VulkanBuffer> textureCoordinatesBuffer;
  std::unique_ptr<VulkanBuffer> indexBuffer;
  std::unique_ptr<VulkanBuffer> materialIndicesBuffer;
  std::unique_ptr<VulkanBuffer> materialBuffer;
  std::unique_ptr<VulkanBuffer> randomInputStateBuffer;
  std::vector<std::unique_ptr<VulkanBuffer> > albedoImageBuffers;
  std::vector<std::unique_ptr<VulkanImage> > albedoImages;
  std::vector<std::unique_ptr<VulkanBuffer> > metallicImageBuffers;
  std::vector<std::unique_ptr<VulkanImage> > metallicImages;
  std::vector<std::unique_ptr<VulkanBuffer> > roughnessImageBuffers;
  std::vector<std::unique_ptr<VulkanImage> > roughnessImages;

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
  VkDescriptorPool descriptorPool{ nullptr };
  VkDescriptorSetLayout descriptorSetLayout{ nullptr };
  VkDescriptorSet descriptorSet{ nullptr };
};

int main(int argc, char* argv[])
{
  if(argc != 3)
  {
    std::cout << " unexpected number of arguments: " << argc << " expecting 3, usage <binary> <renderJobFile> <outputDirectory> " << std::endl;
    return 1;
  }

  std::filesystem::path outputFolder = argv[2];
  if(!std::filesystem::is_directory(outputFolder))
  {
    std::cout << "error given output path " << outputFolder << " is not a directory " << std::endl;
    return 1;
  }

  std::cout << "Loading render job" << std::endl;
  RenderJob job;
  try
  {
    job.loadJSON(argv[1]);
  }
  catch(const std::exception& e)
  {
    std::cout << "exception: " << e.what() << std::endl;
    return 2;
  }

  std::cout << "Setting up path tracer" << std::endl;
  Image3f colorImage, normalImage, albedoImage;
  try
  {
    RaytracerPrototype raytracer;
    raytracer.run(job, colorImage, normalImage, albedoImage);
  }
  catch(const std::exception& e)
  {
    std::cout << "exception: " << e.what() << std::endl;
    return 3;
  }

  try
  {
    postProcessAndSaveToDisk(outputFolder, colorImage, normalImage, albedoImage);
  }
  catch(const std::exception& e)
  {
    std::cout << "exception: " << e.what() << std::endl;
    return 5;
  }

  return 0;
}

