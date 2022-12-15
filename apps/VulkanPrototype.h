
#include <Hasty/Image.h>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <cassert>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include <optional>

inline void checkVkResult(VkResult vkResult, const char* name) {
  if (vkResult != VK_SUCCESS) {
    throw std::runtime_error("failed to " + std::string(name) + ", result = " + std::string(string_VkResult(vkResult)));
  }
}

#define VK_CHECK_RESULT(x)  {  VkResult _result_name123 = ( x ); checkVkResult(_result_name123, #x);  }

inline void printPhysicalDeviceLimits(const VkPhysicalDeviceLimits& limits) {
  std::cout << "  .... PRINTING NOT SUPPORTED .... " << '\n';
}

inline void printPhysicalDeviceSparseProperties(const VkPhysicalDeviceSparseProperties& sparseProperties) {
  std::cout << "  .... PRINTING NOT SUPPORTED .... " << '\n';
}
inline void printPhysicalDevice(const VkPhysicalDevice& physicalDevice) {
  std::cout << " physicalDevice " << physicalDevice << '\n';

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  std::cout << " properties :" << '\n';
  std::cout << "  apiVersion: " << properties.apiVersion << '\n';
  std::cout << "  deviceID: " << properties.deviceID << '\n';
  std::cout << "  deviceName: " << properties.deviceName << '\n';
  std::cout << "  deviceType: " << properties.deviceType << " = " << string_VkPhysicalDeviceType(properties.deviceType) << '\n';
  std::cout << "  driverVersion: " << properties.driverVersion << '\n';
  std::cout << "  limits: ";
  printPhysicalDeviceLimits(properties.limits);
  std::cout << "  pipelineCacheUUID: " << properties.pipelineCacheUUID << '\n';
  std::cout << "  sparseProperties: " << '\n';
  printPhysicalDeviceSparseProperties(properties.sparseProperties);
  std::cout << "  vendorID: " << properties.vendorID << '\n';

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(physicalDevice, &features);
  std::cout << " features " << '\n';
  std::cout << "  robustBufferAccess " << features.robustBufferAccess << '\n';
  std::cout << "  fullDrawIndexUint32 " << features.fullDrawIndexUint32 << '\n';
  std::cout << "  .... " << '\n';
}

inline void printQueueFamilyProperties(const std::string& name, const VkQueueFamilyProperties& familyProperties) {
  std::cout << name << ": \n";
  std::cout << "  queueFlags " << familyProperties.queueFlags << '\n';
  std::cout << "       " << string_VkQueueFlags(familyProperties.queueFlags) << '\n';
  std::cout << "  queueCount " << familyProperties.queueCount << '\n';
  std::cout << "  timestampValidBits " << familyProperties.timestampValidBits << '\n';
  std::cout << "  minImageTransferGranularity  (extent) : " << familyProperties.minImageTransferGranularity.width << ',' << familyProperties.minImageTransferGranularity.height << ',' << familyProperties.minImageTransferGranularity.depth << '\n';
}

class VulkanComputeDeviceAndQueue;

class VulkanBuffer {
  friend class VulkanComputeDeviceAndQueue;
public:
  VulkanBuffer(VulkanComputeDeviceAndQueue& deviceAndQueue, std::size_t bufferSize, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryFlags);

  VulkanBuffer(const VulkanBuffer& b) = delete;
  VulkanBuffer(VulkanBuffer&& b) {
    *this = std::move(b);
  }
  VulkanBuffer& operator=(const VulkanBuffer& b) = delete;
  VulkanBuffer& operator=(VulkanBuffer&& b) {
    this->physicalDevice = b.physicalDevice;
    this->logicalDevice = b.logicalDevice;
    this->buffer = b.buffer;
    this->bufferMemory = b.bufferMemory;
    b.physicalDevice = nullptr;
    b.logicalDevice = nullptr;
    b.buffer = nullptr;
    b.bufferMemory = nullptr;
    return *this;
  }

  ~VulkanBuffer() {
    destroy();
  }

  void loadData(void* pixels, std::size_t imageSize) {
    void* data;
    vkMapMemory(logicalDevice, bufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(logicalDevice, bufferMemory);
  }
  void writeData(void* pixels, std::size_t imageSize) {
    void* data;
    vkMapMemory(logicalDevice, bufferMemory, 0, imageSize, 0, &data);
    memcpy(pixels, data, static_cast<size_t>(imageSize));
    vkUnmapMemory(logicalDevice, bufferMemory);
  }
  void destroy() {
    if (bufferMemory != nullptr) {
      vkFreeMemory(logicalDevice, bufferMemory, nullptr);
      bufferMemory = nullptr;
    }
    if (buffer != nullptr) {
      vkDestroyBuffer(logicalDevice, buffer, nullptr);
      buffer = nullptr;
    }
  }

private:
  VkPhysicalDevice physicalDevice{ nullptr };
  VkDevice logicalDevice{ nullptr };
  VkBuffer buffer{ nullptr };
  VkDeviceMemory bufferMemory{ nullptr };

};
class VulkanImage {
  friend class VulkanComputeDeviceAndQueue;
public:
  VulkanImage(VulkanComputeDeviceAndQueue& deviceAndQueue, uint32_t _width, uint32_t _height, VkFormat _format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties);
  VulkanImage(const VulkanImage& b) = delete;
  VulkanImage(VulkanImage&& b) {
    *this = std::move(b);
  }
  VulkanImage& operator=(const VulkanImage& b) = delete;
  VulkanImage& operator=(VulkanImage&& b) {
    this->physicalDevice = b.physicalDevice;
    this->logicalDevice = b.logicalDevice;
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
    b.physicalDevice = nullptr;
    b.logicalDevice = nullptr;
    b.image = nullptr;
    b.imageMemory = nullptr;
    b.sampler = nullptr;
    b.view = nullptr;

    return *this;
  }

  ~VulkanImage() {
    destroy();
  }

  void updateDescriptor() {
    descriptor.sampler = sampler;
    descriptor.imageView = view;
    descriptor.imageLayout = layout;
  }

  void destroy() {
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
  }


private:
  VkPhysicalDevice physicalDevice{ nullptr };
  VkDevice logicalDevice{ nullptr };
  VkImage image{ nullptr };
  VkDeviceMemory imageMemory{ nullptr };
  uint32_t width;
  uint32_t height;
  VkFormat format;
  VkSampler sampler{ nullptr };
  VkImageLayout layout;
  VkImageView view{ nullptr };
  VkDescriptorImageInfo  descriptor;
  VkPipelineStageFlags stage;
};

class VulkanInstance {
  friend class VulkanComputeDeviceAndQueue;
public:
  VulkanInstance() {

  }
  VulkanInstance(const VulkanInstance& b) = delete;
  VulkanInstance& operator=(const VulkanInstance& b) = delete;

  ~VulkanInstance() {
    destroy();
  }

  void init(bool enableValidationLayers = true) {
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
  void destroy() {
    if (rawInstance != nullptr) {
      vkDestroyInstance(rawInstance, nullptr);
      rawInstance = nullptr;
    }
  }

private:
  VkInstance rawInstance{ nullptr };
};

class VulkanComputeDeviceAndQueue {
  friend class VulkanBuffer;
  friend class VulkanImage;
public:
  VulkanComputeDeviceAndQueue() {

  }
  VulkanComputeDeviceAndQueue(const VulkanComputeDeviceAndQueue& b) = delete;
  VulkanComputeDeviceAndQueue& operator=(const VulkanComputeDeviceAndQueue& b) = delete;

  ~VulkanComputeDeviceAndQueue() {
    destroy();
  }

  void init(VulkanInstance& instance) {
    assert(physicalDevice == nullptr);

    std::vector<const char*> enabledExtensions = { {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
      } };
    uint32_t deviceCount = 0;
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance.rawInstance, &deviceCount, nullptr));

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance.rawInstance, &deviceCount, physicalDevices.data()));

    for (auto availablePhysicalDevice : physicalDevices) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(availablePhysicalDevice, &properties);
      if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        physicalDevice = availablePhysicalDevice;
      }
    }
    for (auto availablePhysicalDevice : physicalDevices) {
      if (availablePhysicalDevice == physicalDevice) std::cout << "*";
      printPhysicalDevice(availablePhysicalDevice);
    }
    if (physicalDevice == nullptr) {
      throw std::runtime_error("no suitable gpu found");
    }

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR physicalDeviceRayTracingPipelineProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };

    VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    prop2.pNext = &physicalDeviceRayTracingPipelineProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &prop2);

    //find correct queue index
    computeQueueFamilyIndex = 0xFFFFFFFF;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    for (uint32_t index = 0; index < queueFamilies.size(); index++) {
      if ((queueFamilies[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        computeQueueFamilyIndex = index;
        break;
      }
    }
    for (uint32_t index = 0; index < queueFamilies.size(); index++) {
      if (index == computeQueueFamilyIndex) std::cout << "*";
      printQueueFamilyProperties("queue " + std::to_string(index), queueFamilies[index]);
    }

    if (computeQueueFamilyIndex >= queueFamilyCount) {
      throw std::runtime_error("no suitable queue found");
    }

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures2 features2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };

    VkDeviceCreateInfo deviceCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
    deviceCreateInfo.pEnabledFeatures = nullptr;
    deviceCreateInfo.pNext = &features2;
    VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice));

    vkGetDeviceQueue(logicalDevice, computeQueueFamilyIndex, 0, &computeQueue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    VK_CHECK_RESULT(vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &computeCommandPool));

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
  }

  void destroy() {
    if (logicalDevice != nullptr) {
      VK_CHECK_RESULT(vkDeviceWaitIdle(logicalDevice));

      if (computeCommandPool != nullptr) {
        for (VkCommandBuffer buffer : submittedSingleTimeCommandBuffer) {
          vkFreeCommandBuffers(logicalDevice, computeCommandPool, 1, &buffer); //TODO use fences and sometimes clear the list
        }
        submittedSingleTimeCommandBuffer.clear();

        vkDestroyCommandPool(logicalDevice, computeCommandPool, nullptr);
        computeCommandPool = nullptr;
      }

      vkDestroyDevice(logicalDevice, nullptr);
      logicalDevice = nullptr;
    }
    physicalDevice = nullptr;
  }

  std::optional<uint32_t> getMemoryTypeIndex(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
      bool memoryTypeSupportedByDeviceForResource = memoryTypeBits & (1 << i);
      bool memoryTypeHasAllProperties = (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;
      if (memoryTypeSupportedByDeviceForResource && memoryTypeHasAllProperties) {
        return i;
      }
    }

    return std::optional<uint32_t>();
  }

  VulkanBuffer allocateHostBuffer(std::size_t imageSize) {
    return VulkanBuffer(*this, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  }

  VulkanImage allocateImage(VkFormat format, std::size_t texWidth, std::size_t texHeight, VkImageUsageFlags usageFlags) {
    return VulkanImage(*this, texWidth, texHeight, format, VK_IMAGE_TILING_OPTIMAL, usageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  VkCommandBuffer beginSingleTimeCommandBuffer() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = computeCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    return commandBuffer;
  }

  void endSingleTimeCommandBuffer(VkCommandBuffer commandBuffer) {
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK_RESULT(vkQueueSubmit(computeQueue, 1, &submitInfo, fence));

    submittedSingleTimeCommandBuffer.emplace_back(commandBuffer);
  }

  // doing an image layout transition while keeping it on the same queue family
  // see https://github.com/Overv/VulkanTutorial/tree/master/code at the time of writing this the license for the code folder is CC0 1.0 Universal.
  void transitionImageLayout(VulkanImage& image, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommandBuffer();

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = image.layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (image.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      barrier.srcAccessMask = 0;
      sourceStage = image.stage;
    }
    else if (image.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      sourceStage = image.stage;
    }
    else if (image.layout == VK_IMAGE_LAYOUT_GENERAL) {
      barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      sourceStage = image.stage;
    }
    else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (newLayout == VK_IMAGE_LAYOUT_GENERAL) {
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
      commandBuffer,
      sourceStage, destinationStage,
      0,
      0, nullptr,
      0, nullptr,
      1, &barrier
    );

    endSingleTimeCommandBuffer(commandBuffer);

    image.layout = newLayout;
    image.stage = destinationStage;
    image.updateDescriptor();
  }

  void copyHostBufferToDeviceImage(VulkanBuffer &buffer, VulkanImage &image) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommandBuffer();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = {
        image.width,
        image.height,
        1
    };

    vkCmdCopyBufferToImage(
      commandBuffer,
      buffer.buffer,
      image.image,
      image.layout,
      1,
      &region
    );
    endSingleTimeCommandBuffer(commandBuffer);
  }

  void waitIdle() {
    vkDeviceWaitIdle(logicalDevice);
  }

 private:
  VkPhysicalDevice physicalDevice{ nullptr };
  VkDevice logicalDevice{ nullptr };
  VkQueue computeQueue{ nullptr };
  VkCommandPool computeCommandPool{ nullptr };
  uint32_t computeQueueFamilyIndex{ 0xFFFFFFFF };
  VkPhysicalDeviceMemoryProperties memoryProperties;
  std::vector<VkCommandBuffer> submittedSingleTimeCommandBuffer;
};

