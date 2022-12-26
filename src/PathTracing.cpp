#include <Hasty/PathTracing.h>

#include <Hasty/Scene.h>
#include <Hasty/Timer.h>
#include <Hasty/Random.h>
#include <Hasty/MIS.h>

#include <xmmintrin.h>
#include <pmmintrin.h>

namespace Hasty
{

void preinitEmbree()
{
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

//estimates L(x,\omega_O), where x is defined by ray hit, and \omega_O by ray
Vec3f estimateRadiance(RenderContext context, LightRayInfo& lightRay, const Ray& ray, Vec3f& normal, Vec3f& albedo, int depth, RayHit* rayhitPtr)
{
  if (depth >= context.renderSettings.maxDepth) return Vec3f::Zero();

  RayHit rayhitLocal;
  RayHit& rayhit = rayhitPtr == nullptr ? rayhitLocal : *rayhitPtr;
  if (rayhitPtr == nullptr) context.scene.rayHit(ray, &rayhit);

  if (hasHitSurface(rayhit))
  {
    if (depth == 0)
    {
      normal = rayhit.interaction.normalShadingDefault;
      albedo = context.scene.getAlbedo(rayhit.interaction);
    }

    Vec3f throughput = beersLaw(lightRay.getTransmittance(), (rayhit.interaction.x - ray.origin()).norm());
    if (context.scene.isSurfaceLight(rayhit))
    {
      return throughput.cwiseProd(context.scene.getEmissionRadiance(-ray.direction(), rayhit));
    }

    constexpr int beta = 2;
    SamplingStrategies strategies(context);
    int numStrategies = strategies.numStrategies();
    SmallVector<float, 3> pdfs(numStrategies);

    // https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting#
    float rouletteQ = 0.0f;
    if (depth >= context.renderSettings.roulleteStartDepth) rouletteQ = context.renderSettings.rouletteQ;

    if (uniform01f(context.rng) < rouletteQ)
    {
      return Vec3f::Zero();
    }
    throughput /= (1.0f - rouletteQ);

    Vec3f estimator = Vec3f::Zero();

    for (int strategy = 0; strategy < numStrategies; strategy++)
    {
      MISSampleResult sampleResult = strategies.sample(context, lightRay, ray, rayhit, strategy);
      pdfs[strategy] = sampleResult.pdfOmega;
      Vec3f throughput2 = throughput.cwiseProd(sampleResult.throughput());
      sampleResult.lightRay.applyWavelength(throughput2);

      if (throughput2 == Vec3f::Zero() || isSelfIntersection(rayhit, sampleResult.rayhit)) continue;

      for (int j = 0; j < numStrategies; j++)
      {
        if (j != strategy) pdfs[j] = strategies.evalPDF(context, rayhit, -ray.direction(), j, sampleResult.ray, sampleResult.rayhit);
      }

      float msiWeight = computeMsiWeightPowerHeuristic<beta>(strategy, pdfs);

      throughput2 *= msiWeight;

      sampleResult.lightRay.updateMedia(context, rayhit, sampleResult.ray.direction());

      Vec3f normal2, albedo2;
      Vec3f Li = estimateRadiance(context, sampleResult.lightRay, sampleResult.ray, normal2, albedo2, depth + 1, &sampleResult.rayhit);
      estimator += Li.cwiseProd(throughput2);
    }

    return estimator;
  }
  else
  {
    Vec3f transmittance = beersLaw(lightRay.getTransmittance(), 1e6f);
    Vec3f Le = context.scene.evalEnvironment(ray);
    Vec3f L = transmittance.cwiseProd(Le);
    normal = Vec3f::Zero();
    albedo = L;
    return L;
  }
}


void renderThread(Image3fAccDoubleBuffer& colorBuffer, Image3fAccDoubleBuffer& normalBuffer, Image3fAccDoubleBuffer& albedoBuffer, RenderJob& job, RenderThreadData& threadData)
{
  Scene& scene = *job.scene;

  RNG rng(threadData.seed);

  Image3f& imageColor = colorBuffer.getWriteBuffer().data;
  imageColor.setZero(job.renderSettings.width, job.renderSettings.height);
  Image3f& imageNormal = normalBuffer.getWriteBuffer().data;
  imageNormal.setZero(job.renderSettings.width, job.renderSettings.height);
  Image3f& imageAlbedo = albedoBuffer.getWriteBuffer().data;
  imageAlbedo.setZero(job.renderSettings.width, job.renderSettings.height);

  RenderContext context(scene, rng, job.renderSettings);

  RenderJobSample sample = job.fetchSample();
  while (!job.stopFlag && sample.idx < job.renderSettings.numSamples)
  {
    int ns = 1;

    std::cout << " sampleIdx " << sample.idx << " numSamples " << colorBuffer.getWriteBuffer().numSamples << '\n';

    HighResTimer timer;

    for (uint32_t i = 0; i < job.renderSettings.width; i++)
    {
      for (uint32_t j = 0; j < job.renderSettings.height; j++)
      {
        Vec2f p0 = Vec2f(float(i), float(j));
        Ray ray = scene.camera.computeRay(rng, p0 + sample.offset, float(job.renderSettings.width), float(job.renderSettings.height));
        Vec3f normal, albedo;
        LightRayInfo rayInfo;

        Vec3f L = estimateRadiance(context, rayInfo, ray, normal, albedo, 0);
        if (L != L)
        {
          std::cout << " ERR NAN" << i << ' ' << j << std::endl;
        }
        else
        {
          imageColor(i, j) += L;
          imageNormal(i, j) += normal;
          imageAlbedo(i, j) += albedo;
          assertFinite(imageAlbedo(i, j));
        }
      }
    }

    colorBuffer.getWriteBuffer().numSamples += ns;
    normalBuffer.getWriteBuffer().numSamples += ns;
    albedoBuffer.getWriteBuffer().numSamples += ns;
    colorBuffer.copyBuffer();
    normalBuffer.copyBuffer();
    albedoBuffer.copyBuffer();

    sample = job.fetchSample();

    std::cout << (ns / timer.seconds()) << "samples per second" << std::endl;

  }
  std::cout << " stopping " << '\n';
  threadData.stoppedFlag = true;
}

}
