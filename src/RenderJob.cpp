#include <Hasty/RenderJob.h>

#include <Hasty/Scene.h>

#include <Hasty/Json.h>

namespace Hasty
{

void from_json(const nlohmann::json& j, RenderSettings& settings)
{
  json_get_optional(settings.rouletteStartDepth, j, "roulette_start_depth", 1);
  json_get_optional(settings.rouletteQ, j, "roulette_q", 0.3f);
  json_get_optional(settings.width, j, "width", uint32_t(720));
  json_get_optional(settings.height, j, "height", uint32_t(720));
  json_get_optional(settings.numThreads, j, "num_threads", uint32_t(4));
  json_get_optional(settings.maxDepth, j, "max_depth", 30);
  json_get_optional(settings.numSamples, j, "num_samples", 32);
  json_get_optional(settings.useMIS, j, "use_mis", true);
}
void RenderJob::loadJSON(std::filesystem::path jsonFilepath)
{
  std::filesystem::path folder = jsonFilepath.parent_path();
  nlohmann::json jsonData = readJSON(jsonFilepath);
  from_json(jsonData["settings"], renderSettings);
  std::filesystem::path scenePath = folder / jsonData["scene"].get<std::string>();
  this->scene = std::make_unique<Scene>(scenePath);
}

}
