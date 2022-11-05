#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Image.h>

#include <memory>
#include <string>
#include <filesystem>

namespace Hasty
{

class RayHit;
struct SurfaceInteraction;

template<typename ResultType>
class ITextureMap
{
public:
  virtual ~ITextureMap() {}
  virtual ResultType evaluate(const SurfaceInteraction& interaction) = 0;
};

using ITextureMap1f = ITextureMap<float>;
using ITextureMap3f = ITextureMap<Vec3f>;

template<typename ResultType>
class ConstantTexture : public ITextureMap<ResultType>
{
public:
  ConstantTexture(const ResultType& c) :m_color(c) {}
  virtual ResultType evaluate(const SurfaceInteraction& interaction)
  {
    return m_color;
  }

private:
  ResultType m_color;
};

using ConstantTexture1f = ConstantTexture<float>;
using ConstantTexture3f = ConstantTexture<Vec3f>;

struct SamplingOptions
{
  enum class AddressMode
  {
    Repeat, ClampToEdge
  };

  AddressMode modeU = AddressMode::Repeat;
  AddressMode modeV = AddressMode::Repeat;
  
};


template<typename ResultType>
class TextureTraits
{

};

template<typename ResultType>
class Texture;

template<>
class TextureTraits<float>
{
public:
  static std::unique_ptr<Texture<float> > loadTexture(const std::filesystem::path& filename, ColorSpace colorSpaceInFile);
  static float NoUV()
  {
    return float(-1.0f);
  }
};
template<>
class TextureTraits<Vec3f>
{
public:
  static std::unique_ptr<Texture<Vec3f> > loadTexture(const std::filesystem::path& filename, ColorSpace colorSpaceInFile);
  static Vec3f NoUV()
  {
    return Vec3f(1.0f, 0.0f, 0.7f);
  }
};

template<typename ResultType>
class Texture : public ITextureMap<ResultType>
{
public:

  // expects data to already be linear
  Texture(Image<ResultType> image)
    :m_image(std::move(image))
  {
  }
  ~Texture()
  {
  }
  
  virtual ResultType evaluate(const SurfaceInteraction& interaction);
  ResultType evaluateUV(const Vec2f& uv) const
  {
    float u, oneMinusV;
    switch (options.modeU)
    {
    case SamplingOptions::AddressMode::Repeat: u = fmod1p1(uv[0]); break;
    case SamplingOptions::AddressMode::ClampToEdge: u = clamp(uv[0], 0.0f, 1.0f - std::numeric_limits<float>::epsilon()); break;
    default: throw std::runtime_error("u mode not implemented");
    }
    switch (options.modeV)
    {
    case SamplingOptions::AddressMode::Repeat: oneMinusV = fmod1p1(1.0f - uv[1]); break;  // putting one minus here, so result is definetly in [0, 1) // our coordinate system here is : 0,0 is bottom left of image (last row), 1,1 is top r ight
    case SamplingOptions::AddressMode::ClampToEdge: oneMinusV = clamp(1.0f - uv[1], 0.0f, 1.0f - std::numeric_limits<float>::epsilon()); break;
    default: throw std::runtime_error("v mode not implemented");
    }

    int col = static_cast<int>(u * m_image.getWidth());
    int row = static_cast<int>(oneMinusV * m_image.getHeight());
    ResultType color = m_image(col, row);
    return color;
  }

  static std::unique_ptr<Texture<ResultType> > loadTexture(const std::filesystem::path& filename, ColorSpace colorSpaceInFile)
  {
    return TextureTraits<ResultType>::loadTexture(filename, colorSpaceInFile);
  }

  const Image<ResultType>& getImage() const { return m_image; }
  
  SamplingOptions options;
private:
  Image<ResultType> m_image;
};

using Texture1f = Texture<float>;
using Texture3f = Texture<Vec3f>;



}
