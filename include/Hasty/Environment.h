#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Texture.h>
#include <Hasty/Random.h>

#include <memory>
#include <random>

namespace Hasty
{

class Background
{
public:
  virtual ~Background(){}
  virtual Vec3f evalEmission(const Ray& ray) = 0;
  //virtual float powerPerArea() = 0;
  virtual bool isZero() const = 0;
};
class BackgroundColor : public Background
{
public:
  BackgroundColor(const Vec3f &c):color(c){}

  Vec3f evalEmission(const Ray& ray)
  {
    return color;
  }
  //float powerPerArea()
  //{
  //  return 4.0f * float(Hasty::Pi) * color.norm();
  //}

  bool isZero() const
  {
    return color == Vec3f::Zero();
  }
  
private:
  Vec3f color;

};

class ImageDistribution
{
public:
  
  void precompute(Image1f img)
  {
    notNormalizedPdf = std::move(img);
    auto b = begin(notNormalizedPdf);
    auto e = end(notNormalizedPdf);
    distribution = std::make_unique< std::discrete_distribution<uint32_t> >(b, e);
    valueSum = std::accumulate(b, e, 0.f);
  }

  Vec2f sample(RNG& rng, float *density) const
  {
    uint32_t idx = (*distribution)(rng);
    uint32_t x = idx % notNormalizedPdf.getWidth();
    uint32_t y = idx / notNormalizedPdf.getWidth();
    float heightF = float(notNormalizedPdf.getHeight());
    float widthF = float(notNormalizedPdf.getWidth());
    float xi0 = uniform01f(rng);
    float xi1 = uniform01f(rng);
    float left = x / widthF;
    float right = (x + 1.0f) / widthF;
    float lower = (notNormalizedPdf.getHeight() - (y + 1.0f)) / heightF;
    float upper = (notNormalizedPdf.getHeight() - y) / heightF;
    float u = (1.0f - xi0) * left + xi0 * right;
    float v = (1.0f - xi1) * lower + xi1 * upper;
    float pxy = notNormalizedPdf(x, y) / valueSum;
    assert(pxy != 0.0f);
    (*density) = widthF * heightF * pxy ; //  p(u,v) = p(u,v | x, y) * p(x, y), p(u,v | x, y) = p(xi0) / abs(det(jacobian0 )) p(xi1) / abs(det(jacobian1))
    return Vec2f(u, v);
  }

  float evalDensity(const Vec2f &uv) const
  {
    float heightF = float(notNormalizedPdf.getHeight());
    float widthF = float(notNormalizedPdf.getWidth());
    uint32_t x = std::min(std::size_t(uv[0] * notNormalizedPdf.getWidth()), notNormalizedPdf.getWidth() - 1);
    uint32_t y = std::min(std::size_t((1.0f - uv[1]) * notNormalizedPdf.getHeight()), notNormalizedPdf.getHeight() - 1);
    return widthF * heightF * notNormalizedPdf(x, y) / valueSum;
  }
  
  std::unique_ptr<std::discrete_distribution<uint32_t> > distribution;
  Image1f notNormalizedPdf;
  float valueSum;
};
class EnvironmentTexture : public Background
{
public:
  EnvironmentTexture(const std::filesystem::path& path)
  {
    this->load(path);
  }

  void load(const std::filesystem::path& path)
  {
    texture = Texture3f::loadTexture(path, ColorSpace::Linear);
    texture->options.modeU = SamplingOptions::AddressMode::Repeat;
    texture->options.modeV = SamplingOptions::AddressMode::ClampToEdge;
    float heightF = texture->getImage().getHeight();
    Image1f imgNorm = transform<float>(texture->getImage(), [heightF](int x, int y, const auto& pixel) {
      float theta = ( (y + 0.5f) / heightF) * float(Hasty::Pi);
      return pixel.norm() * std::sin(theta);
      });
    distribution.precompute(std::move(imgNorm));

  }
  
  Vec2f getUVFromSphericalCoordinates(const Vec2f& c) const
  {
    Vec2f uv;
    uv[0] = c[1] * float(0.5 * Hasty::InvPi);
    uv[1] = 1.0f -  c[0] * float( Hasty::InvPi);
    return uv;
  }
  Vec2f getSphericalCoordinatesFromUV(const Vec2f& uv) const
  {
    Vec2f c;
    c[1] = uv[0] * 2.0f * float(Hasty::Pi);
    c[0] = (1.0f - uv[1]) * float(Hasty::Pi);
    return c;
  }
  Vec2f getSphericalCoordinates(const Vec3f& direction) const
  {
    return SphericalCoordinates::FromDirection(direction);
  }
  Vec3f evalEmission(const Ray& ray)
  {
    Vec2f c = getSphericalCoordinates(ray.direction());
    Vec2f uv = getUVFromSphericalCoordinates(c);
    return texture->evaluateUV(uv);
  }

  Vec3f sample(RNG &rng, float *density) const
  {
    SphericalCoordinates::ToDirectionResult scResult;
    do
    {
      Vec2f uv = distribution.sample(rng, density);
      Vec2f c = getSphericalCoordinatesFromUV(uv);
      scResult = SphericalCoordinates::ToDirection(c);
    } while (scResult.sinTheta == 0.0f);
    (*density) /= (2.0f * float(Hasty::Pi) * float(Hasty::Pi) * scResult.sinTheta);
    return scResult.direction;
  }
  bool isZero() const
  {
    return false;
  }

  float evalSamplingDensity(const Vec3f &dir)
  {
    Vec2f c = SphericalCoordinates::FromDirection(dir);
    Vec2f uv = getUVFromSphericalCoordinates(c);
    float sinTheta = std::sin(c[0]);
    float density = distribution.evalDensity(uv) / std::max(1e-7f, 2.0f * float(Hasty::Pi * Hasty::Pi) * sinTheta);
    return density;
  }

private:
  std::unique_ptr<Texture3f> texture;
  ImageDistribution distribution;
};

}
