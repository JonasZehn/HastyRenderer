#include "VulkanPrototype.h"

#include <Hasty/File.h>
#include <VulkanInitializers.hpp> 

VulkanBuffer::VulkanBuffer(const std::shared_ptr<VulkanComputeDeviceAndQueue>& _deviceAndQueue, std::size_t bufferSize, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryFlags)
{
  assert(buffer == nullptr);

  deviceAndQueue = _deviceAndQueue;
  VkPhysicalDevice physicalDevice = deviceAndQueue->physicalDevice;
  VkDevice logicalDevice = deviceAndQueue->logicalDevice;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;
  bufferInfo.usage = usageFlags;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VK_CHECK_RESULT(vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  std::optional<uint32_t> memoryTypeIndex = deviceAndQueue->getMemoryTypeIndex(memRequirements.memoryTypeBits, memoryFlags);
  if (!memoryTypeIndex.has_value()) {
    throw std::runtime_error("no memory type found");
  }
  allocInfo.memoryTypeIndex = memoryTypeIndex.value();
  VK_CHECK_RESULT(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory));

  VK_CHECK_RESULT(vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0));
}
VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& b) {
  this->deviceAndQueue = b.deviceAndQueue;
  this->buffer = b.buffer;
  this->bufferMemory = b.bufferMemory;
  b.deviceAndQueue = nullptr;
  b.buffer = nullptr;
  b.bufferMemory = nullptr;
  return *this;
}
void VulkanBuffer::loadData(void* pixels, std::size_t imageSize) {
  VkDevice logicalDevice = deviceAndQueue->logicalDevice;

  void* data;
  vkMapMemory(logicalDevice, bufferMemory, 0, imageSize, 0, &data);
  memcpy(data, pixels, static_cast<size_t>(imageSize));
  vkUnmapMemory(logicalDevice, bufferMemory);
}
void VulkanBuffer::writeData(void* pixels, std::size_t imageSize) {
  VkDevice logicalDevice = deviceAndQueue->logicalDevice;

  void* data;
  vkMapMemory(logicalDevice, bufferMemory, 0, imageSize, 0, &data);
  memcpy(pixels, data, static_cast<size_t>(imageSize));
  vkUnmapMemory(logicalDevice, bufferMemory);
}
void VulkanBuffer::destroy() {
  if (deviceAndQueue) {
    VkDevice logicalDevice = deviceAndQueue->logicalDevice;

    if (bufferMemory != nullptr) {
      vkFreeMemory(logicalDevice, bufferMemory, nullptr);
      bufferMemory = nullptr;
    }
    if (buffer != nullptr) {
      vkDestroyBuffer(logicalDevice, buffer, nullptr);
      buffer = nullptr;
    }

    deviceAndQueue = nullptr;
  }
}
VulkanImage::VulkanImage(const std::shared_ptr<VulkanComputeDeviceAndQueue>& _deviceAndQueue, uint32_t _width, uint32_t _height, VkFormat _format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties) {
  assert(image == nullptr);

  deviceAndQueue = _deviceAndQueue;
  VkPhysicalDevice physicalDevice = deviceAndQueue->getPhysicalDevice();
  VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

  width = _width;
  height = _height;
  format = _format;
  layout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImageCreateInfo imageCreateInfo{};
  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
  imageCreateInfo.format = format;
  imageCreateInfo.mipLevels = 1;
  imageCreateInfo.arrayLayers = 1;
  imageCreateInfo.extent = { width, height, 1 };
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCreateInfo.tiling = tiling;
  imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageCreateInfo.initialLayout = layout;
  imageCreateInfo.usage = usage;

  VK_CHECK_RESULT(vkCreateImage(logicalDevice, &imageCreateInfo, nullptr, &image));

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  std::optional<uint32_t> memoryTypeIndex = deviceAndQueue->getMemoryTypeIndex(memRequirements.memoryTypeBits, properties);
  if (!memoryTypeIndex.has_value()) {
    throw std::runtime_error("no memory type found");
  }
  allocInfo.memoryTypeIndex = memoryTypeIndex.value();

  VK_CHECK_RESULT(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory));

  vkBindImageMemory(logicalDevice, image, imageMemory, 0);

  // https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanTexture.cpp
  // Create a default sampler
  VkSamplerCreateInfo samplerCreateInfo = {};
  samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.mipLodBias = 0.0f;
  samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
  samplerCreateInfo.minLod = 0.0f;
  samplerCreateInfo.maxLod = 0.0f;
  samplerCreateInfo.maxAnisotropy = 1.0f;
  samplerCreateInfo.anisotropyEnable = false;
  samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  VK_CHECK_RESULT(vkCreateSampler(logicalDevice, &samplerCreateInfo, nullptr, &sampler));

  VkImageViewCreateInfo viewCreateInfo = {};
  viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewCreateInfo.format = format;
  viewCreateInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
  viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
  viewCreateInfo.subresourceRange.levelCount = 1;
  viewCreateInfo.image = image;
  VK_CHECK_RESULT(vkCreateImageView(logicalDevice, &viewCreateInfo, nullptr, &view));

  updateDescriptor();

  stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
}
VulkanImage& VulkanImage::operator=(VulkanImage&& b) {
  this->deviceAndQueue = b.deviceAndQueue;
  this->image = b.image;
  this->imageMemory = b.imageMemory;
  this->width = b.width;
  this->height = b.height;
  this->format = b.format;
  this->sampler = b.sampler;
  this->layout = b.layout;
  this->view = b.view;
  this->descriptor = b.descriptor;
  this->stage = b.stage;
  b.deviceAndQueue = nullptr;
  b.image = nullptr;
  b.imageMemory = nullptr;
  b.sampler = nullptr;
  b.view = nullptr;

  return *this;
}

VulkanImage::~VulkanImage() {
  destroy();
}

void VulkanImage::updateDescriptor() {
  descriptor.sampler = sampler;
  descriptor.imageView = view;
  descriptor.imageLayout = layout;
}

void VulkanImage::destroy() {
  if (deviceAndQueue) {
    VkDevice logicalDevice = deviceAndQueue->logicalDevice;
    if (sampler != nullptr) {
      vkDestroySampler(logicalDevice, sampler, nullptr);
      sampler = nullptr;
    }
    if (view != nullptr) {
      vkDestroyImageView(logicalDevice, view, nullptr);
      view = nullptr;
    }

    if (imageMemory != nullptr) {
      vkFreeMemory(logicalDevice, imageMemory, nullptr);
      imageMemory = nullptr;
    }
    if (image != nullptr) {
      vkDestroyImage(logicalDevice, image, nullptr);
      image = nullptr;
    }
    deviceAndQueue = nullptr;
  }
}
void VulkanInstance::init(bool enableValidationLayers) {
  assert(rawInstance == nullptr);

  std::vector<const char*> validationLayers;
  if (enableValidationLayers) {
    validationLayers.push_back("VK_LAYER_KHRONOS_validation");
  }

  std::vector<const char*> requiredInstanceExtensions;

  VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
  appInfo.pApplicationName = "HastyPrototype";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "Hasty";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo createInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  createInfo.pApplicationInfo = &appInfo;

  createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredInstanceExtensions.size());
  createInfo.ppEnabledExtensionNames = requiredInstanceExtensions.data();

  createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
  createInfo.ppEnabledLayerNames = validationLayers.data();

  VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &rawInstance));
}
void VulkanInstance::destroy() {
  if (rawInstance != nullptr) {
    vkDestroyInstance(rawInstance, nullptr);
    rawInstance = nullptr;
  }
}
VulkanFence::VulkanFence(const std::shared_ptr<VulkanComputeDeviceAndQueue>& _deviceAndQueue) {
  deviceAndQueue = _deviceAndQueue;
  VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

  VkFenceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  vkCreateFence(logicalDevice, &createInfo, nullptr, &fence);
}
VulkanFence& VulkanFence::operator=(VulkanFence&& b) {
  this->deviceAndQueue = b.deviceAndQueue;
  this->fence = b.fence;
  b.deviceAndQueue = nullptr;
  b.fence = nullptr;

  return *this;
}
void VulkanFence::destroy() {
  if (deviceAndQueue) {
    VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();
    if (fence != nullptr) {
      vkDestroyFence(logicalDevice, fence, nullptr);
      fence = nullptr;
    }
    deviceAndQueue = nullptr;
  }
}

class LaplacianFilterExample {
public:
  LaplacianFilterExample(){}
  LaplacianFilterExample(const LaplacianFilterExample& b) = delete;
  LaplacianFilterExample& operator=(const LaplacianFilterExample& b) = delete;
  ~LaplacianFilterExample() {
    destroy();
  }

private:
  void buildComputeCommandBuffer(uint32_t width, uint32_t height)
  {
    VkQueue queue = deviceAndQueue->getQueue();

    // Flush the queue if we're rebuilding the command buffer after a pipeline change to ensure it's not currently in use
    VK_CHECK_RESULT(vkQueueWaitIdle(queue));

    VkCommandBufferBeginInfo commandBufferBeginInfo = vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

    uint32_t layoutSizeX = 16;
    uint32_t layoutSizeY = 16;
    uint32_t groupCountX = (width + layoutSizeX - 1) / layoutSizeX;
    uint32_t groupCountY = (height + layoutSizeY - 1) / layoutSizeY;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }

public:
  void preprareCompute(const std::shared_ptr<VulkanComputeDeviceAndQueue> &_deviceAndQueue, VulkanImage& srcImage, VulkanImage& dstImage) {
    assert(!deviceAndQueue);
    assert(srcImage.getWidth() == dstImage.getWidth());
    assert(srcImage.getHeight() == dstImage.getHeight());

    deviceAndQueue = _deviceAndQueue;

    VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();

    // setupDescriptorPool
    std::vector<VkDescriptorPoolSize> poolSizes = {
      // Graphics pipelines image samplers for displaying compute output image
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2),
      // Compute pipelines uses a storage image for image reads and writes
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2),
    };
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);
    VK_CHECK_RESULT(vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool));

    // Create compute pipeline
    // Compute pipelines are created separate from graphics pipelines even if they use the same queue

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
      // Binding 0: Input image (read-only)
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0),
      // Binding 1: Output image (write)
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logicalDevice, &descriptorLayout, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

    VK_CHECK_RESULT(vkCreatePipelineLayout(logicalDevice, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

    VK_CHECK_RESULT(vkAllocateDescriptorSets(logicalDevice, &allocInfo, &descriptorSet));

    std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &srcImage.getDescriptor()),
      vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &dstImage.getDescriptor())
    };
    vkUpdateDescriptorSets(logicalDevice, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);

    // Create compute shader pipelines
    VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(pipelineLayout, 0);

    // One pipeline for each effect
    std::vector<char> shaderBinary = Hasty::readFile(HASTY_SHADER_PATH / "laplaceShader.comp.spv");
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
    buildComputeCommandBuffer(srcImage.getWidth(), srcImage.getHeight());
  }
  void run() {
    VkQueue queue = deviceAndQueue->getQueue();

    // Wait for rendering finished
    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // Submit compute commands
    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &commandBuffer;
    computeSubmitInfo.waitSemaphoreCount = 0;
    computeSubmitInfo.pWaitSemaphores = VK_NULL_HANDLE;
    computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores = &semaphore;
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
  }
  void destroy() {
    if (deviceAndQueue) {
      VkDevice logicalDevice = deviceAndQueue->getLogicalDevice();
      if (semaphore != nullptr) {
        vkDestroySemaphore(logicalDevice, semaphore, nullptr);
        semaphore = nullptr;
      }
      if (commandBuffer != nullptr) {
        vkFreeCommandBuffers(logicalDevice, deviceAndQueue->getCommandPool(), 1, &commandBuffer);
        commandBuffer = nullptr;
      }
      if (pipeline != nullptr) {
        vkDestroyPipeline(logicalDevice, pipeline, nullptr);
        pipeline = nullptr;
      }
      if (descriptorSetLayout != nullptr) {
        vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
        descriptorSetLayout = nullptr;
      }
      if (pipelineLayout != nullptr) {
        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
        pipelineLayout = nullptr;
      }
      if (descriptorPool != nullptr) {
        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
        descriptorPool = nullptr;
      }
      descriptorSet = nullptr;

      deviceAndQueue = std::shared_ptr<VulkanComputeDeviceAndQueue>();
    }
  }
private:
  VulkanShaderModule shader;
  VkPipeline pipeline{ nullptr };
  VkPipelineLayout pipelineLayout{ nullptr };

  VkSemaphore semaphore{ nullptr };
  VkCommandBuffer commandBuffer{ nullptr };
  std::shared_ptr<VulkanComputeDeviceAndQueue> deviceAndQueue;
  VkDescriptorPool descriptorPool{ nullptr };
  VkDescriptorSetLayout descriptorSetLayout{ nullptr };
  VkDescriptorSet descriptorSet{ nullptr };
};


VulkanBuffer allocateHostBuffer(std::shared_ptr<VulkanComputeDeviceAndQueue> deviceAndQueue, std::size_t imageSize) {
  return VulkanBuffer(deviceAndQueue, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

VulkanImage allocateDeviceImage(std::shared_ptr<VulkanComputeDeviceAndQueue> deviceAndQueue, VkFormat format, std::size_t texWidth, std::size_t texHeight, VkImageUsageFlags usageFlags) {
  return VulkanImage(deviceAndQueue, texWidth, texHeight, format, VK_IMAGE_TILING_OPTIMAL, usageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

int main()
{
  try {
    VulkanInstance instance;
    instance.init();
    std::shared_ptr<VulkanComputeDeviceAndQueue> deviceAndQueue = std::make_shared<VulkanComputeDeviceAndQueue>();
    deviceAndQueue->init(instance);

    Hasty::Image3f imageIn = Hasty::readImage3f("./glass.jpg");
    Hasty::Image4f image = Hasty::addAlphaChannel(imageIn);
    std::size_t imageByteCount = 4 * sizeof(float) * image.size();
    VulkanBuffer buffer = allocateHostBuffer(deviceAndQueue, imageByteCount);
    buffer.loadData(image.data(), imageByteCount);
    VulkanImage gpuSrcImage = allocateDeviceImage(deviceAndQueue, VK_FORMAT_R32G32B32A32_SFLOAT, image.getWidth(), image.getHeight(), VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    deviceAndQueue->transitionImageLayout(gpuSrcImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    deviceAndQueue->copyBufferToImage(buffer, gpuSrcImage);

    VulkanImage gpuDstImage = allocateDeviceImage(deviceAndQueue, VK_FORMAT_R32G32B32A32_SFLOAT, image.getWidth(), image.getHeight(), VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    deviceAndQueue->transitionImageLayout(gpuSrcImage, VK_IMAGE_LAYOUT_GENERAL);
    deviceAndQueue->transitionImageLayout(gpuDstImage, VK_IMAGE_LAYOUT_GENERAL);

    LaplacianFilterExample computeExample;
    computeExample.preprareCompute(deviceAndQueue, gpuSrcImage, gpuDstImage);
    computeExample.run();

    deviceAndQueue->transitionImageLayout(gpuDstImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VulkanFence fence(deviceAndQueue);
    deviceAndQueue->copyImageToBuffer(gpuDstImage, buffer, fence);
    deviceAndQueue->waitForFence(fence);

    buffer.writeData(image.data(), imageByteCount);

    Hasty::writeEXR(image, "output.exr");

    deviceAndQueue->waitDeviceIdle();

    computeExample.destroy();

    deviceAndQueue->waitDeviceIdle();
  }
  catch (std::runtime_error& e) {
    std::cout << " error " << e.what() << '\n';
    return -1;
  }

  return 0;
}
