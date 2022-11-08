#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Image.h>
#include <Hasty/PathTracing.h>
#include <Hasty/Timer.h>
#include <Hasty/HashCells.h>

namespace Hasty
{

struct PMRenderJobPixel
{
  PMRenderJobPixel() {}
  PMRenderJobPixel(const PMRenderJobPixel& p)
  {
    radius = p.radius;
    N = p.N;
    Ne = p.Ne;
    flux = p.flux;
    LSum = p.LSum;
    Nd = p.Nd;
  }
  std::mutex mutex;
  float radius = -1.0f;
  float N = 0.0f;
  float Ne = 0.0f;
  Vec3f flux = Vec3f::Zero();
  Vec3f LSum = Vec3f::Zero();
  int Nd = 0;
};

class PMRenderJob
{
public:
  void loadJSON(std::filesystem::path jsonFilepath);

  RenderSettings renderSettings;
  HaltonSequence2 pixelRNG;
  std::shared_ptr<Scene> scene;
  std::atomic<int> sampleIdxGenerator = 0;
  std::atomic<bool> stopFlag = false;


  float lock(int i, int j, float defaultRadius)
  {
    auto& pixel = pixels[i + j * pixelsWidth];
    pixel.mutex.lock();
    if (pixel.radius == -1.0f)
    {
      pixel.radius = defaultRadius;
    }
    return pixel.radius;
  }
  
  Vec3f unlock(int i, int j)
  {
    auto& pixel = pixels[i + j * pixelsWidth];
    Vec3f result = pixel.LSum /(1e-7f + float(pixel.Nd));
    pixel.mutex.unlock();
    return result;
  }
  Vec3f unlockAndUpdateStatistic(int i, int j, const Vec3f& flux, const Vec3f& L, int M, int NeInI, float* outRadius)
  {
    float alpha = 0.4f;

    auto& pixel = pixels[i + j * pixelsWidth];
    pixel.LSum += L;
    pixel.Nd += 1;
    Vec3f Lflux = Vec3f::Zero();
    if (pixel.radius < 0.0f)
    {
      if (*outRadius < 0.0f)
      {
        assert(M == 0);
      }
      else
      {
        pixel.radius = *outRadius;
        pixel.N += alpha * M;
        pixel.Ne += NeInI;
        pixel.flux = flux;
        Lflux = pixel.Ne == 0 ? Vec3f::Zero() : pixel.flux / (pixel.radius * pixel.radius * float(Pi) * pixel.Ne);
      }

    }
    else
    {
      float rold = pixel.radius;
      float Ni = pixel.N;
      if (Ni + M > 0)
      {
        pixel.radius *= std::sqrt((Ni + alpha * M) / (Ni + M));
      }
      pixel.N += alpha * M;
      pixel.Ne += NeInI;
      pixel.flux = (pixel.flux + flux) * (pixel.radius * pixel.radius) / (rold * rold);
      Lflux = pixel.Ne == 0 ? Vec3f::Zero() : pixel.flux / (pixel.radius * pixel.radius * float(Pi) * pixel.Ne);
      (*outRadius) = pixel.radius;
    }
    
    Vec3f result = Lflux + pixel.LSum / float(pixel.Nd);
    pixel.mutex.unlock();
    return result;
  }

  RenderJobSample fetchSample()
  {
    auto lock = std::scoped_lock(m_sampleMutex);
    RenderJobSample sample;
    sample.idx = sampleIdxGenerator++;
    sample.offset = pixelRNG();
    return sample;
  }

private:
  std::vector<PMRenderJobPixel> pixels;
  int pixelsWidth;

  std::mutex m_sampleMutex;
};


void renderPhotonThread(Image3fAccDoubleBuffer& colorBuffer, Image3fAccDoubleBuffer& normalBuffer, Image3fAccDoubleBuffer& albedoBuffer, PMRenderJob& job, RenderThreadData& threadData);

}
