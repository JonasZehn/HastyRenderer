#include <Hasty/MIS.h>

namespace Hasty
{

SampleResult BRDFSamplingStrategy::sample(RenderContext context, LightRayInfo& lightRay, const Ray& ray, const RayHit& rayhit)
{
  SampleResult result;

  bool adjoint = false;
  bool outside;
  Vec3f direction = context.scene.sampleBRDF(context, lightRay, rayhit, adjoint, &result.throughputDiffuse, &result.throughputSpecular, &result.pdfOmega, &outside);
  if (result.pdfOmega == 0.0f)
  {
    return result;
  }
  result.ray = context.scene.constructRay(rayhit.interaction, direction, outside);
  context.scene.rayHit(result.ray, &result.rayhit);
  return result;
}
float BRDFSamplingStrategy::evalPDF(RenderContext context, const RayHit& rayhit, const Vec3f& woGlobal, const Ray& ray2, const RayHit& rayhit2)
{
  return context.scene.evaluateSamplePDF(rayhit, ray2.direction());
}

SampleResult LightSamplingStrategy::sample(RenderContext context, LightRayInfo& lightRay, const Ray& ray, const RayHit& rayhit)
{
  SampleResult result;

  bool lightVisible;
  result.ray = context.scene.sampleLightRayFromStartPoint(context.rng, rayhit.interaction, &result.pdfOmega, &result.rayhit, &lightVisible);
  
  if (lightVisible && result.pdfOmega != 0.0f)
  {
    bool adjoint = false;
    MaterialEvalResult evalResult = context.scene.evaluteBRDF(rayhit, -ray.direction(), result.ray.direction(), lightRay.getOutsideIOR(context, rayhit), adjoint, flags);
    float m = std::abs(result.ray.direction().dot(rayhit.interaction.normalGeometric)) / result.pdfOmega;
    result.throughputDiffuse = evalResult.fDiffuse * m; // using evalResult.normalShading here, because it can be flipped, creates some "artifacts"
    result.throughputSpecular = evalResult.fSpecular * m; // using evalResult.normalShading here, because it can be flipped, creates some "artifacts"
    //(*throughput) = evalResult.f * std::max(0.0f, ray2->direction().dot(evalResult.normalShading)) / (*pdfOmega); // using evalResult.normalShading here, because it can be flipped, creates some "artifacts"
  }

  return result;
}
float LightSamplingStrategy::evalPDF(RenderContext context, const RayHit& rayhit, const Vec3f& woGlobal, const Ray& ray2, const RayHit& rayhit2)
{
  return context.scene.evalLightRayFromStartPointDensity(rayhit.interaction, ray2, rayhit2);
}

}
