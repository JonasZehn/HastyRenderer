#include <Hasty/Vulkan.h>

namespace Hasty {

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

  VkMemoryAllocateFlagsInfo flagInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;

  if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    flagInfo.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocInfo.pNext = &flagInfo;
  }

  std::optional<uint32_t> memoryTypeIndex = deviceAndQueue->getMemoryTypeIndex(memRequirements.memoryTypeBits, memoryFlags);
  if (!memoryTypeIndex.has_value()) {
    throw std::runtime_error("no memory type found");
  }
  allocInfo.memoryTypeIndex = memoryTypeIndex.value();
  VK_CHECK_RESULT(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory));

  VK_CHECK_RESULT(vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0));
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
VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& b) {
  this->deviceAndQueue = b.deviceAndQueue;
  this->buffer = b.buffer;
  this->bufferMemory = b.bufferMemory;
  b.deviceAndQueue = nullptr;
  b.buffer = nullptr;
  b.bufferMemory = nullptr;
  return *this;
}
void VulkanBuffer::write(void* src, std::size_t byteCount) {
  VkDevice logicalDevice = deviceAndQueue->logicalDevice;

  void* dst;
  vkMapMemory(logicalDevice, bufferMemory, 0, byteCount, 0, &dst);
  memcpy(dst, src, static_cast<size_t>(byteCount));
  vkUnmapMemory(logicalDevice, bufferMemory);
}
void VulkanBuffer::read(void* dst, std::size_t byteCount) {
  VkDevice logicalDevice = deviceAndQueue->logicalDevice;

  void* src;
  vkMapMemory(logicalDevice, bufferMemory, 0, byteCount, 0, &src);
  memcpy(dst, src, static_cast<size_t>(byteCount));
  vkUnmapMemory(logicalDevice, bufferMemory);
}
VkDeviceAddress VulkanBuffer::getDeviceAddress(VkDevice logicalDevice) {
  VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
  bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  bufferDeviceAddressInfo.pNext = nullptr;
  bufferDeviceAddressInfo.buffer = this->buffer;
  return vkGetBufferDeviceAddress(logicalDevice, &bufferDeviceAddressInfo);
}
VkDescriptorBufferInfo VulkanBuffer::descriptorBufferInfo() {
  VkDescriptorBufferInfo result{};
  result.buffer = this->buffer;
  result.range = VK_WHOLE_SIZE;
  return result;
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
  appInfo.apiVersion = VK_API_VERSION_1_2;

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

}
