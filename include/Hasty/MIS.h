#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Scene.h>
#include <Hasty/PathTracing.h>

namespace Hasty
{

template<int betaN, typename T>
float computeMsiWeightPowerHeuristic(int index, const T& pdfs)
{
  float s = 0.0f;
  for(int i = 0; i < pdfs.size(); i++)
  {
    s += powci<betaN>(pdfs[i]);
  }
  float result = powci<betaN>(pdfs[index]) / s;
  assertFinite(result);
  return result;
}


class MISSampleResult
{
public:
  Ray ray;
  LightRayInfo lightRay;
  RayHit rayhit;
  Vec3f throughputDiffuse;
  Vec3f throughputConcentrated;
  float pdfOmega;

  Vec3f throughput() const
  {
    return throughputDiffuse + throughputConcentrated;
  }
};

class BRDFSamplingStrategy
{
public:

  MISSampleResult sample(RenderContext context, const LightRayInfo& lightRay, const Ray& ray, const RayHit& rayhit);
  float evalPDF(RenderContext context, const RayHit& rayhit, const LightRayInfo& lightRay, const Vec3f& woGlobal, const Ray& ray2, const RayHit& rayhit2);
};

class LightSamplingStrategy
{
public:
  ShaderEvalFlag flags = ShaderEvalFlag::ALL;
  MISSampleResult sample(RenderContext context, const LightRayInfo& lightRay, const Ray& ray, const RayHit& rayhit);
  float evalPDF(RenderContext context, const RayHit& rayhit, const Vec3f& woGlobal, const Ray& ray2, const RayHit& rayhit2);
};

// doesn't sample  a direction, SAMPLES AN ENDPOINT! (otherwise MIS is more costly than it needs to be)
class SamplingStrategies
{
public:

  SamplingStrategies(RenderContext context);

  LightSamplingStrategy& getLightStrategy() { return lightStrategy; }
  int numStrategies()
  {
    return (lightStrategyIndex != -1 ? 1 : 0) + (brdfStrategyIndex != -1 ? 1 : 0);
  }
  bool isLightStrategy(int strategy)
  {
    return lightStrategyIndex == strategy;
  }

  // samples a POINT x2 and computes an according probability density and visibility
  MISSampleResult sample(RenderContext context, const LightRayInfo& lightRay, const Ray& ray, const RayHit& rayhit, int strategy)
  {
    if(isLightStrategy(strategy))
    {
      return lightStrategy.sample(context, lightRay, ray, rayhit);
    }
    else
    {
      assert(strategy == brdfStrategyIndex);
      return brdfStrategy.sample(context, lightRay, ray, rayhit);
    }
  }

  float evalPDF(RenderContext context, const RayHit& rayhit, const LightRayInfo& lightRay, const Vec3f& woGlobal, int strategy, const Ray& ray2, const RayHit& rayhit2)
  {
    if(isLightStrategy(strategy))
    {
      return lightStrategy.evalPDF(context, rayhit, woGlobal, ray2, rayhit2);
    }
    else
    {
      assert(strategy == brdfStrategyIndex);
      return brdfStrategy.evalPDF(context, rayhit, lightRay, woGlobal, ray2, rayhit2);
    }
  }
private:
  BRDFSamplingStrategy brdfStrategy;
  LightSamplingStrategy lightStrategy;
  int brdfStrategyIndex = -1;
  int lightStrategyIndex = -1;
};


}
