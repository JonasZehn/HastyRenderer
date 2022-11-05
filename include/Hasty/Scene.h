#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Camera.h>
#include <Hasty/Random.h>
#include <Hasty/BRDF.h>
#include <Hasty/Environment.h>

#include <tiny_obj_loader.h>
#include <embree3/rtcore.h>

#include <memory>
#include <filesystem>
#include <random>
#include <optional>

namespace Hasty
{

struct SurfaceInteraction
{
  uint32_t geomID;
  uint32_t primID;
  float h;
  Vec3f x;
  Vec3f normalGeometric;
  Vec3f normalShadingDefault;
  int32_t materialId;
  std::optional<Vec2f> uv;
  Vec3f barycentricCoordinates;
};

class RayHit
{
public:
  RTCRayHit rtc;
  SurfaceInteraction interaction;

  Vec3f wFrom() const
  {
    return Vec3f(-rtc.ray.dir_x, -rtc.ray.dir_y, -rtc.ray.dir_z);
  }

};
inline bool hasHitSurface(const RayHit& rayhit)
{
  return rayhit.rtc.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}
inline bool hasHitSurface(const RayHit& rayhit, unsigned int geomID, unsigned int primID)
{
  return rayhit.rtc.hit.geomID == geomID && rayhit.rtc.hit.primID == primID;
}


class Scene
{
public:
  Camera camera;
  std::unique_ptr<Background> background;

  Scene(std::filesystem::path inputfilePath);
  ~Scene();

  SurfaceInteraction getInteraction(unsigned int geomID, unsigned int primID, const Vec3f& barycentricCoordinates, const Vec3f& geomNormal);
  Vec3f constructRayX(const SurfaceInteraction& interaction, bool outside);
  Ray constructRay(const SurfaceInteraction& interaction, const Vec3f& direction, bool outside);
  Ray constructRayEnd(const SurfaceInteraction& interaction, const Vec3f& end, bool outside);
  void rayHit(const Ray& ray, RayHit* rayhit);
protected:
  BXDF& getBXDF(const SurfaceInteraction& interaction);
  const BXDF& getBXDF(const SurfaceInteraction& interaction) const;
public:
  MaterialEvalResult evaluteBRDF(const RayHit& hit, const Vec3f& wo, const Vec3f& wi, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag flags = ShaderEvalFlag::ALL);
  Vec3f sampleBRDF(RenderContext& context, LightRayInfo& lightRay, const RayHit& rayhit, bool adjoint, Vec3f* throughputDiffuse, Vec3f* throughputSpecular, float* pDensity, bool *outside);
  Vec3f sampleBRDFSpecular(RenderContext& context, LightRayInfo& lightRay, const RayHit& rayhit, bool adjoint, Vec3f* throughputDiffuse, Vec3f* throughputSpecular, float* pDensity, bool *outside);
  float evaluateSamplePDF(const RayHit& rayhit, const Vec3f& direction2);
  bool hasBRDFDiffuseLobe(const RayHit& rayhit);
  float getIORInside(const RayHit& rayhit, int wavelength);
protected:
  const tinyobj::material_t& getMaterial(unsigned int geomId, unsigned int primId) const;
  const tinyobj::material_t& getMaterial(const RayHit& hit) const;
public:
  Vec3f getEmissionRadiance(const Vec3f& wo, unsigned int geomId, unsigned int primId) const;
  Vec3f getEmissionRadiance(const Vec3f& wo, const RayHit& hit) const;
  Vec3f getAlbedo(const SurfaceInteraction& interaction) const;
  Vec3f evalEnvironment(const Ray& ray);
  bool hasLight() const;
  bool hasSurfaceLight() const;
  bool isSurfaceLight(const RayHit& hit) const;
  bool isInfiniteAreaLight(const RayHit& hit) const;
  std::array<Vec3f, 3> collectTriangle(std::size_t geomID, std::size_t primID) const;
  std::array<Vec3f, 3> collectTriangleNormals(std::size_t geomID, std::size_t primID) const;
  SurfaceInteraction sampleSurfaceLightPosition(RNG& rng, float* pDensity);
  Ray sampleLightRay(RNG& rng, Vec3f* flux);
  Ray sampleLightRayFromStartPoint(RNG& rng, const SurfaceInteraction& point, float* pDensity, RayHit *rayhit, bool *lightVisible);
  float evalLightRayFromStartPointDensity(const SurfaceInteraction& point, const Ray& ray2, const RayHit& rayhit2);
  float computeSurfaceLightProbabilityDensity(const RayHit& hit) const;
  std::string getObjectName(const RayHit& hit);

protected:

  RTCDevice m_device;
  RTCScene m_embreeScene;

  tinyobj::ObjReader reader;
  std::unique_ptr<std::discrete_distribution<uint32_t> > lightDistribution;
  std::vector<std::array<std::size_t, 2> > lightTriangles;
  std::vector<std::unique_ptr<BXDF> > m_brdfs;
  float totalLightArea;
};

}