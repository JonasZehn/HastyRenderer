#pragma once

#include <Hasty/Random.h>
#include <Hasty/VMath.h>
#include <Hasty/SmallVector.h>

namespace Hasty
{

class RenderContext;
class RayHit;

class LightRayInfoMedium
{
public:
  LightRayInfoMedium()
    : geomID(-1), transmittance(Vec3f::Ones()), indexOfRefraction(1.0f)
  {
  }
  LightRayInfoMedium(int geomID, const Vec3f& t, float indexOfRefraction)
    : geomID(geomID), transmittance(t), indexOfRefraction(indexOfRefraction)
  {
  }

  Vec3f transmittance;
  float indexOfRefraction;
  int geomID;
};

class LightRayInfo
{
public:
  LightRayInfo()
    :wavelength(-1)
  {
    transmittance = Vec3f::Ones();
    probabilityChange = 1.0f;
  }

  const Vec3f& getTransmittance() const { return transmittance; }

  void updateMedia(const RenderContext& context, const RayHit& hit, const Vec3f& wi);
  int getWavelength(RNG& rng, const Vec3f& probabilities);
  void applyWavelength(Vec3f& throughput);
  float getOutsideIOR(const RenderContext& context, const RayHit& rayhit);

  int wavelength;

private:
  void recompute();
  SmallVector<LightRayInfoMedium, 3> m_media;
  Vec3f transmittance;
  float probabilityChange;
};

}
