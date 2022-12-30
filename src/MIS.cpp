#include <Hasty/MIS.h>

namespace Hasty
{

MISSampleResult BRDFSamplingStrategy::sample(RenderContext context, const LightRayInfo& lightRay, const Ray& ray, const RayHit& rayhit)
{
  MISSampleResult result;

  bool adjoint = false;
  SampleResult sampleResult = context.scene.sampleBRDF(context, lightRay, rayhit, adjoint);
  if (sampleResult.pdfOmega == 0.0f)
  {
    result.throughputDiffuse = Vec3f::Zero();
    result.throughputConcentrated = Vec3f::Zero();
    result.pdfOmega = 0.0f;
    return result;
  }
  result.ray = context.scene.constructRay(rayhit.interaction, sampleResult.direction, sampleResult.outside);
  result.lightRay = sampleResult.lightRay;
  result.throughputDiffuse = sampleResult.throughputDiffuse;
  result.throughputConcentrated = sampleResult.throughputConcentrated;
  result.pdfOmega = sampleResult.pdfOmega;
  context.scene.rayHit(result.ray, &result.rayhit);
  return result;
}
float BRDFSamplingStrategy::evalPDF(RenderContext context, const RayHit& rayhit, const LightRayInfo& lightRay, const Vec3f& woGlobal, const Ray& ray2, const RayHit& rayhit2)
{
  return context.scene.evaluateSamplePDF(context, rayhit, lightRay, ray2.direction());
}

MISSampleResult LightSamplingStrategy::sample(RenderContext context, const LightRayInfo& lightRay, const Ray& ray, const RayHit& rayhit)
{
  MISSampleResult result;
  result.lightRay = lightRay;

  bool lightVisible;
  result.ray = context.scene.sampleLightRayFromStartPoint(context.rng, rayhit.interaction, &result.pdfOmega, &result.rayhit, &lightVisible);
  
  if (!lightVisible || result.pdfOmega == 0.0f)
  {
    result.throughputDiffuse = Vec3f::Zero();
    result.throughputConcentrated = Vec3f::Zero();
    result.pdfOmega = 0.0f;
    return result;
  }
  bool adjoint = false;
  MaterialEvalResult evalResult = context.scene.evaluteBRDF(rayhit, -ray.direction(), result.ray.direction(), result.lightRay.getOutsideIOR(context, rayhit), adjoint, flags);
  float m = std::abs(dot(result.ray.direction(), rayhit.interaction.normalGeometric)) / result.pdfOmega;
  result.throughputDiffuse = evalResult.fDiffuse * m; // using evalResult.normalShading here, because it can be flipped, creates some "artifacts"
  result.throughputConcentrated = evalResult.fConcentrated * m; // using evalResult.normalShading here, because it can be flipped, creates some "artifacts"

  return result;
}
float LightSamplingStrategy::evalPDF(RenderContext context, const RayHit& rayhit, const Vec3f& woGlobal, const Ray& ray2, const RayHit& rayhit2)
{
  return context.scene.evalLightRayFromStartPointDensity(rayhit.interaction, ray2, rayhit2);
}

SamplingStrategies::SamplingStrategies(RenderContext context)
{
  if (context.scene.hasLight() && context.renderSettings.useMIS)
  {
    lightStrategyIndex = 0;
    brdfStrategyIndex = 1;
  }
  else
  {
    lightStrategyIndex = -1;
    brdfStrategyIndex = 0;
  }
}
}
