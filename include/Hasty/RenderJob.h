#pragma once

#include <Hasty/Random.h>

#include <cstdint>
#include <mutex>
#include <filesystem>

namespace Hasty
{

class Scene;

class RenderSettings
{
public:

  RenderSettings() {}

  int roulleteStartDepth = 1;
  float rouletteQ = 0.3f;
  uint32_t width = 720;
  uint32_t height = 720;
  uint32_t numThreads = 4;
  int maxDepth = 30;
  int numSamples = 32;
  bool useMIS = true;
  float exposure = 0.0f; // relative exposure where 0 returns "raw" value
};
void from_json(const nlohmann::json& j, RenderSettings& settings);


class RenderJobSample
{
public:
  int idx;
  Vec2f offset;
};


class RenderJob
{
public:
  void loadJSON(std::filesystem::path jsonFilepath);

  RenderSettings renderSettings;
  HaltonSequence2 pixelRNG;
  std::shared_ptr<Scene> scene;
  std::atomic<int> sampleIdxGenerator = 0;
  std::atomic<bool> stopFlag = false;

  RenderJobSample fetchSample()
  {
    auto lock = std::scoped_lock(m_sampleMutex);
    RenderJobSample sample;
    sample.idx = sampleIdxGenerator++;
    sample.offset = pixelRNG();
    return sample;
  }

private:
  std::mutex m_sampleMutex;
};

}
