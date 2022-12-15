#include "VulkanPrototype.h"

VulkanBuffer::VulkanBuffer(VulkanComputeDeviceAndQueue& deviceAndQueue, std::size_t bufferSize, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryFlags)
{
  assert(buffer == nullptr);

  physicalDevice = deviceAndQueue.physicalDevice;
  logicalDevice = deviceAndQueue.logicalDevice;

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
  std::optional<uint32_t> memoryTypeIndex = deviceAndQueue.getMemoryTypeIndex(memRequirements.memoryTypeBits, memoryFlags);
  if (!memoryTypeIndex.has_value()) {
    throw std::runtime_error("no memory type found");
  }
  allocInfo.memoryTypeIndex = memoryTypeIndex.value();
  VK_CHECK_RESULT(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory));

  VK_CHECK_RESULT(vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0));
}
VulkanImage::VulkanImage(VulkanComputeDeviceAndQueue& deviceAndQueue, uint32_t _width, uint32_t _height, VkFormat _format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties) {
  assert(image == nullptr);

  physicalDevice = deviceAndQueue.physicalDevice;
  logicalDevice = deviceAndQueue.logicalDevice;
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
  std::optional<uint32_t> memoryTypeIndex = deviceAndQueue.getMemoryTypeIndex(memRequirements.memoryTypeBits, properties);
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

int main()
{
  try {
    VulkanInstance instance;
    instance.init();
    VulkanComputeDeviceAndQueue deviceAndQueue;
    deviceAndQueue.init(instance);

    Hasty::Image3f imageIn = Hasty::readImage3f("./glass.jpg");
    Hasty::Image4f image = Hasty::addAlphaChannel(imageIn);
    std::size_t imageByteCount = 4 * sizeof(float) * image.size();
    VulkanBuffer buffer = deviceAndQueue.allocateHostBuffer(imageByteCount);
    buffer.loadData(image.data(), imageByteCount);
    VulkanImage gpuSrcImage = deviceAndQueue.allocateImage(VK_FORMAT_R32G32B32A32_SFLOAT, image.getWidth(), image.getHeight(), VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    deviceAndQueue.transitionImageLayout(gpuSrcImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    deviceAndQueue.copyHostBufferToDeviceImage(buffer, gpuSrcImage);

    VulkanImage gpuDstImage = deviceAndQueue.allocateImage(VK_FORMAT_R32G32B32A32_SFLOAT, image.getWidth(), image.getHeight(), VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    deviceAndQueue.transitionImageLayout(gpuSrcImage, VK_IMAGE_LAYOUT_GENERAL);
    deviceAndQueue.transitionImageLayout(gpuDstImage, VK_IMAGE_LAYOUT_GENERAL);

    deviceAndQueue.waitIdle();
  }
  catch (std::runtime_error& e) {
    std::cout << " error " << e.what() << '\n';
    return -1;
  }

  return 0;
}