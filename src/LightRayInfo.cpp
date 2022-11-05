#include <Hasty/LightRayInfo.h>

#include <Hasty/PathTracing.h>

namespace Hasty
{

void LightRayInfo::updateMedia(const RenderContext& context, const RayHit& hit, const Vec3f& wTo)
{
  //std::cout << " hit.normalGeometric.dot(wi) " << hit.normalGeometric.dot(wi) << std::endl;
  float ndotFrom = hit.interaction.normalGeometric.dot(hit.wFrom());
  float ndotTo = hit.interaction.normalGeometric.dot(wTo);
  if (ndotFrom >= 0.0f && ndotTo < 0.0f)
  { //entering medium
    //std::cout << " entering medium " << hit.rtc.hit.geomID << std::endl;
    Vec3f t = context.scene.getAlbedo(hit.interaction);
    float indexOfRefractionInside = context.scene.getIORInside(hit, this->wavelength);
    m_media.emplace_back(LightRayInfoMedium(hit.rtc.hit.geomID, t, indexOfRefractionInside));
    recompute();
  }
  else if (ndotFrom < 0.0f && ndotTo >= 0.0f)
  { // leaving medium
    //std::cout << " leaving medium " << hit.rtc.hit.geomID << std::endl;
    int leaveMed = ((int)hit.rtc.hit.geomID);
    int medIdx = -1;
    for (int i = 0; i < m_media.size(); i++)
    {
      if (m_media[i].geomID == leaveMed)
      {
        medIdx = i;
        break;
      }
    }
    //std::cout << " removing " << leaveMed << " from slot " << medIdx << std::endl;
    if (medIdx == -1)
    {
      //std::cout << "remove: warning medium not found " << " ndotFrom " << ndotFrom << " ndotTo " << ndotTo << " name: " << context.scene.getObjectName(hit) << std::endl;
    }
    else
    {
      m_media.remove(medIdx);
      recompute();
    }
  }
}
void LightRayInfo::recompute()
{
  for (int i = 0; i < m_media.size(); i++)
  {
    if (m_media[i].geomID != -1)
    {
      this->transmittance = m_media[i].transmittance;
      return;
    }
  }
  this->transmittance = Vec3f::Ones();
}

int LightRayInfo::getWavelength(RNG& rng, const Vec3f& weights)
{
  if (wavelength == -1)
  {
    Vec3f probabilities = weights / weights.sum();
    float xi = rng.uniform01f();
    if (xi < probabilities[0])
    {
      wavelength = 0;
      probabilityChange = probabilities[0];
    }
    else if (xi < probabilities[0] + probabilities[1])
    {
      wavelength = 1;
      probabilityChange = probabilities[1];
    }
    else
    {
      wavelength = 2;
      probabilityChange = probabilities[2];
    }
  }
  else
  {
    probabilityChange = 1.0f;
  }
  return wavelength;
}
float LightRayInfo::getOutsideIOR(const RenderContext& context, const RayHit& rayhit)
{
  // this structure stores the current index of refraction; we have to deal with multiple overlapping Manifold shapes, if they don't overlap then we will fetch the medium from the world and geomID becomes -1
  //  if they overlap we will assume that the first material we hit is the medium (this is an inconsistent view of the world but cheaper) and when we leave the other object, we keep the medium...
  if (rayhit.wFrom().dot(rayhit.interaction.normalGeometric) < 0.0f)
  {
    //coming from the inside of an object

    int leaveMed = ((int)rayhit.rtc.hit.geomID);
    int medIdx = -1;
    for (int i = 0; i < m_media.size(); i++)
    {
      if (m_media[i].geomID == leaveMed)
      {
        medIdx = i;
        break;
      }
    }
    if (medIdx == -1)
    {
      //std::cout << "warning medium not found, dot " << rayhit.wFrom().dot(rayhit.normalGeometric) << " leaving " << leaveMed << " normal " << rayhit.normalGeometric << " wfrom " << rayhit.wFrom() << std::endl;
      // this can also be a non manifold object from the other side
      return 1.0f;
    }
    else
    {
      if (m_media.size() == 1) return 1.0f;
      else
      {
        if (medIdx == 0) return m_media[1].indexOfRefraction;
        else return m_media[0].indexOfRefraction;
      }
    }
  }
  else
  {
    // coming from the outside of an object
    if (m_media.size() == 0) return 1.0f;
    else
    {
      return m_media[0].indexOfRefraction;
    }
  }
}
void LightRayInfo::applyWavelength(Vec3f& throughput)
{
  if (wavelength != -1)
  {
    throughput[wavelength] *= (1.0f / probabilityChange);
    throughput[(wavelength + 1) % 3] = 0.0f;
    throughput[(wavelength + 2) % 3] = 0.0f;
    probabilityChange = 1.0f;
  }
}

}