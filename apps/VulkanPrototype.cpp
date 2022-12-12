
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

void checkVkResult(VkResult vkResult, const char* name) {
  if (vkResult != VK_SUCCESS) {
    throw std::runtime_error("failed to " + std::string(name) + ", result = " + std::string(string_VkResult(vkResult)));
  }
}

#define VK_CHECK_RESULT(x)  {  VkResult _result_name123 = ( x  ); checkVkResult(_result_name123, #x);  }

void printPhysicalDeviceLimits(const VkPhysicalDeviceLimits& limits) {
  std::cout << "  .... PRINTING NOT SUPPORTED .... " << '\n';
}

void printPhysicalDeviceSparseProperties(const VkPhysicalDeviceSparseProperties& sparseProperties) {
  std::cout << "  .... PRINTING NOT SUPPORTED .... " << '\n';
}
void printPhysicalDevice(const VkPhysicalDevice& physicalDevice) {
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
  printPhysicalDeviceSparseProperties(properties.sparseProperties);
  std::cout << "  vendorID: " << properties.vendorID << '\n';

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(physicalDevice, &features);
  std::cout << " features " << '\n';
  std::cout << "  robustBufferAccess " << features.robustBufferAccess << '\n';
  std::cout << "  fullDrawIndexUint32 " << features.fullDrawIndexUint32 << '\n';
  std::cout << "  .... " << '\n';
}

void printQueueFamilyProperties(const std::string& name, const VkQueueFamilyProperties& familyProperties) {
  std::cout << name << ": \n";
  std::cout << "  queueFlags " << familyProperties.queueFlags << '\n';
  std::cout << "       " << string_VkQueueFlags(familyProperties.queueFlags) << '\n';
  std::cout << "  queueCount " << familyProperties.queueCount << '\n';
  std::cout << "  timestampValidBits " << familyProperties.timestampValidBits << '\n';
  std::cout << "  minImageTransferGranularity  (extent) : " << familyProperties.minImageTransferGranularity.width << ',' << familyProperties.minImageTransferGranularity.height << ',' << familyProperties.minImageTransferGranularity.depth << '\n';
}

class VulkanInstance {
public:
  VulkanInstance() {

  }

  void init(bool enableValidationLayers = true) {
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
    vkDestroyInstance(rawInstance, nullptr);
  }

  VkInstance rawInstance = nullptr;
};

class VulkanComputeDeviceAndQueue {
public:
  VulkanComputeDeviceAndQueue() {

  }
  void init(VulkanInstance& instance) {

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

    physicalDevice = nullptr;
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

  }
  void destroy() {
    VK_CHECK_RESULT(vkDeviceWaitIdle(logicalDevice));
    vkDestroyCommandPool(logicalDevice, computeCommandPool, nullptr);
    vkDestroyDevice(logicalDevice, nullptr);
  }

  VkPhysicalDevice physicalDevice;
  VkDevice logicalDevice;
  VkQueue computeQueue;
  VkCommandPool computeCommandPool;
  uint32_t computeQueueFamilyIndex;
};

int main()
{
  try {
    VulkanInstance instance;
    instance.init();
    VulkanComputeDeviceAndQueue deviceAndQueue;
    deviceAndQueue.init(instance);

    deviceAndQueue.destroy();
    instance.destroy();
  }
  catch (std::runtime_error& e) {
    std::cout << " error " << e.what() << '\n';
    return -1;
  }

  return 0;
}