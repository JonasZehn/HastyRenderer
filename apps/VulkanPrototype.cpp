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
  }
  catch (std::runtime_error& e) {
    std::cout << " error " << e.what() << '\n';
    return -1;
  }

  return 0;
}