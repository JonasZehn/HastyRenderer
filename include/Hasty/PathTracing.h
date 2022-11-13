#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Image.h>
#include <Hasty/Scene.h>
#include <Hasty/RenderJob.h>

struct RTCRayHit;

namespace Hasty
{

void preinitEmbree();

class RenderThreadData
{
public:
  unsigned int seed;
  std::atomic<bool> stoppedFlag;

  RenderThreadData(unsigned int _seed)
    :seed(_seed), stoppedFlag(false)
  {

  }
};

class RenderContext
{
public:
  RenderContext(Scene& scene, RNG& rng, RenderSettings& renderSettings)
    : scene(scene), rng(rng), renderSettings(renderSettings)
  {

  }
  Scene& scene;
  RNG& rng;
  RenderSettings& renderSettings;
};

inline bool isSelfIntersection(const RayHit& rayhit, const RayHit& rayhit2)
{
  return rayhit.rtc.hit.geomID == rayhit2.rtc.hit.geomID &&
    rayhit.rtc.hit.primID == rayhit2.rtc.hit.primID;
}
inline Vec3f beersLaw(const Vec3f& transmittance, float d)
{
  if (transmittance == Vec3f::Ones()) return Vec3f::Ones(); // hot code path

  Vec3f result;
  float dRef = 1.0f;
  float invdRef = 1.0f / dRef;
  for (int i = 0; i < 3; i++)
  {
    result[i] = transmittance[i] <= 0.0f ? 0.0f : std::pow(transmittance[i], invdRef * d); //  std::exp(std::logf(transmittance[i]) * (invdRef * d))
  }
  assertFinite(result);
  return result;
}
//estimates L(x,\omega_O), where x is defined by ray hit, and \omega_O by ray
Vec3f estimateRadiance(RenderContext context, LightRayInfo& lightRay, const Ray& ray, Vec3f& normal, Vec3f& albedo, int depth, RayHit* rayhitPtr = nullptr);

void renderThread(Image3fAccDoubleBuffer& colorBuffer, Image3fAccDoubleBuffer& normalBuffer, Image3fAccDoubleBuffer& albedoBuffer, RenderJob& job, RenderThreadData& data);

}
