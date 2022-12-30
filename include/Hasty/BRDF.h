#pragma once

#include <Hasty/Random.h>
#include <Hasty/VMath.h>
#include <Hasty/LightRayInfo.h>
#include <Hasty/Texture.h>
#include <Hasty/Sample.h>

#include <functional>

namespace Hasty
{

typedef std::function<float()> OutsideIORFunctor;
class LightRayInfo;
class RenderContext;
class RayHit;
struct SurfaceInteraction;

bool computeRefractionDirection(const Vec3f& wi, const Vec3f& normal, float indexOfRefraction_i, float indexOfRefraction_t, Vec3f* wt);
float fresnelDielectric(float cos_i, float cos_t, float indexOfRefraction_i, float indexOfRefraction_t);

Vec3f sampleDGGX(RNG& rng, const Vec3f& normal, float alpha, const Vec3f& dir1, float* pDensity);
float sampleDGGXDensity(const Vec3f& normal, float alpha, const Vec3f& dir1, const Vec3f& dir2);

float DGGX(float nh, float th, float bh, float alpha_t, float alpha_b);

Vec3f sampleGGXVNDFGlobal(RNG& rng, const Vec3f& normal, float alpha_t, float alpha_b, const Vec3f& tangent, const Vec3f& bitangent, const Vec3f& dir1, float* pDensity);
float sampleGGXVNDFGlobalDensity(const Vec3f& normal, float alpha_t, float alpha_b, const Vec3f& tangent, const Vec3f& bitangent, const Vec3f& dir1, const Vec3f& dir2);

enum class ShaderEvalFlag
{
  NONE = 0,
  DIFFUSE = 1,
  CONCENTRATED = 2,
  ALL = DIFFUSE | CONCENTRATED
};
inline bool has(ShaderEvalFlag flags, ShaderEvalFlag single)
{
  assert(single == ShaderEvalFlag::DIFFUSE || single == ShaderEvalFlag::CONCENTRATED);
  return (static_cast<int>(flags) & static_cast<int>(single)) != 0;
}

struct MaterialEvalResult
{
  MaterialEvalResult(const Vec3f& fDiffuse, const Vec3f& fConcentrated, const Vec3f& n)
    :fDiffuse(fDiffuse), fConcentrated(fConcentrated), normalShading(n)
  {

  }
  Vec3f fDiffuse;
  Vec3f fConcentrated;
  Vec3f normalShading;
};

struct SampleResult
{
  bool outside;
  Vec3f direction;
  LightRayInfo lightRay;
  Vec3f throughputDiffuse;
  Vec3f throughputConcentrated;
  float pdfOmega;

  Vec3f throughput() const
  {
    return throughputDiffuse + throughputConcentrated;
  }
};

class BXDF
{
public:
  virtual ~BXDF() {}
  virtual Vec3f getAlbedo(const SurfaceInteraction& interaction) const = 0;
  virtual MaterialEvalResult evaluate(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag evalFlag) = 0;

  virtual SampleResult sample(RNG& rng, const SurfaceInteraction& interaction, const LightRayInfo& lightRay, const Vec3f& wOut, OutsideIORFunctor getOutsideIOR, bool adjoint, ShaderEvalFlag evalFlag) = 0;
  virtual float evaluateSamplePDF(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float outsideIOR) = 0;

  virtual bool hasDiffuseLobe(const SurfaceInteraction& interaction) = 0;
  virtual float getIndexOfRefraction(int wavelength) const = 0;
};

// https://en.wikipedia.org/wiki/Fused_quartz
inline float fusedQuartzIndexOfRefraction(float wavelengthMicroMeter)
{
  float wavelengthMicroMeterSquare = wavelengthMicroMeter;
  float a = 0.6961663f * wavelengthMicroMeterSquare / (wavelengthMicroMeterSquare - 0.0684043f * 0.0684043f);
  float b = 0.4079426f * wavelengthMicroMeterSquare / (wavelengthMicroMeterSquare - 0.1162414f * 0.1162414f);
  float c = 0.8974794f * wavelengthMicroMeterSquare / (wavelengthMicroMeterSquare - 9.896161f * 9.896161f);
  return std::sqrt(1.0f + a + b + c);
}

class PrincipledBRDF final : public BXDF
{
public:

  PrincipledBRDF(std::unique_ptr<ITextureMap3f> albedo, std::unique_ptr<ITextureMap1f> roughness, std::unique_ptr<ITextureMap1f> metallic, float specular, float indexOfRefraction, std::unique_ptr<ITextureMap3f> normalMap, float anisotropy, float transmission, bool varyIOR);
  ~PrincipledBRDF();

  float getIndexOfRefraction(int wavelength) const;
  const ITextureMap3f& getAlbedo() const { return *albedo; }
  Vec3f getAlbedo(const SurfaceInteraction& interaction) const { return albedo->evaluate(interaction); }

  const ITextureMap1f& getRoughness() const { return *roughness; }
  const ITextureMap1f& getMetallic() const { return *metallic; }

  float getSpecular() const { return specular; }

  // returns normal in direction of wo
  Vec3f getShadingNormal(const SurfaceInteraction& interaction, const Vec3f& wo, float dotNgWo);
  float computeAlpha(float roughness);

  struct ProbabilityResult
  {
    bool noStrategy;
    float pDiffuseStrategy;
    float pSpecularStrategy;
    float pRefractiveStrategy;
    bool refractionPossible;
    float FDaccurate;
    float cosT;
  };
  ProbabilityResult computeProbability(float metallicHit, float specularHit, float transmissionHit, float cos_o, float indexOfRefraction_o, float indexOfRefraction_t, ShaderEvalFlag evalFlag);
  void computeAnisotropyParameters(const SurfaceInteraction& interaction, const Vec3f& normalShading, float alpha, float& alpha_t, float& alpha_b, Vec3f& tangent, Vec3f& bitangent);
  MaterialEvalResult evaluate(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag evalFlag);
  SampleResult sample(RNG& rng, const SurfaceInteraction& interaction, const LightRayInfo& lightRay, const Vec3f& wOut, OutsideIORFunctor getOutsideIOR, bool adjoint, ShaderEvalFlag evalFlag);
  float evaluateSamplePDF(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float outsideIOR);
  bool hasDiffuseLobe(const SurfaceInteraction& interaction);

private:
  std::unique_ptr<ITextureMap3f> albedo;
  std::unique_ptr<ITextureMap1f> roughness;
  std::unique_ptr<ITextureMap1f> metallic;
  float specular;
  float indexOfRefraction;
  std::unique_ptr<ITextureMap3f> normalMap;
  float anisotropy;
  float transmission;
  bool varyIOR;
};

}
