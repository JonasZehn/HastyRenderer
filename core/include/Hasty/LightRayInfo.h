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
  int getWavelength() const { return wavelength; }
  int sampleOrGetWavelength(RNG& rng, const Vec3f& sampleWeights, bool sample, float& sampleProbability);
  float getOutsideIOR(const RenderContext& context, const RayHit& rayhit) const;

private:
  int wavelength;

  void recompute();
  SmallVector<LightRayInfoMedium, 4> m_media;
  Vec3f transmittance;
  float probabilityChange;
};

typedef int32_t Wavelength;

}
