
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

#define VK_CHECK_RESULT(x)  {  VkResult _result_nam123 = ( x  ); checkVkResult(_result_nam123, #x);  }

class VulkanInstance {
public:
  VulkanInstance() {

  }

  void init() {
    const bool enableValidationLayers = true;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "HastyPrototype";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Hasty";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    std::vector<const char*> requiredInstanceExtensions;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    createInfo.enabledExtensionCount = (uint32_t)requiredInstanceExtensions.size();
    createInfo.ppEnabledExtensionNames = requiredInstanceExtensions.data();

    uint32_t layerCount;
    VK_CHECK_RESULT(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));

    std::vector<VkLayerProperties> availableLayers(layerCount);
    VK_CHECK_RESULT(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &rawInstance));
  }
  void destroy() {
    vkDestroyInstance(rawInstance, nullptr);
  }

  VkInstance rawInstance = nullptr;
};


int main()
{
  try {
    VulkanInstance instance;
    instance.init();

    instance.destroy();

  }
  catch (std::runtime_error& e) {
    std::cout << " error " << e.what() << '\n';
  }

  return 0;
}