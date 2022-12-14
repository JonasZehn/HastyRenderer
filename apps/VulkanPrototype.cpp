#include "VulkanPrototype.h"

int main()
{
  try {
    VulkanInstance instance;
    instance.init();
    VulkanComputeDeviceAndQueue deviceAndQueue;
    deviceAndQueue.init(instance);
  }
  catch (std::runtime_error& e) {
    std::cout << " error " << e.what() << '\n';
    return -1;
  }

  return 0;
}