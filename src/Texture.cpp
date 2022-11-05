#include <Hasty/Texture.h>

#include <Hasty/Scene.h>
#include <Hasty/Image.h>

namespace Hasty
{

template<typename ResultType>
ResultType Texture<ResultType>::evaluate(const SurfaceInteraction& interaction)
{
  if (!interaction.uv.has_value())
  {
    return TextureTraits<ResultType>::NoUV();
  }
  const Vec2f& uv = interaction.uv.value();
  return this->evaluateUV(uv);
}


std::unique_ptr<Texture<float> > TextureTraits<float>::loadTexture(const std::filesystem::path& filename, ColorSpace colorSpaceInFile)
{
  Image1f image;
  if (filename.extension() == ".png")
  {
    image = readPNG1(filename);
    image /= 255.f; // convert from [0, 255] to [0, 1]
  }
  else
  {
    throw std::runtime_error("unsupported extension of " + filename.string());
  }
  
  switch (colorSpaceInFile)
  {
  case ColorSpace::Linear: break;
  case ColorSpace::sRGB: clampAndPowInplace(image, 0.0f, std::numeric_limits<float>::max(), 2.2f); break;
  default: throw std::runtime_error("unsupported color space conversion");
  }
  
  std::unique_ptr<Texture<float> > result = std::make_unique<Texture<float> >(std::move(image));
  return result;
}
std::unique_ptr<Texture<Vec3f> > TextureTraits<Vec3f>::loadTexture(const std::filesystem::path& filename, ColorSpace colorSpaceInFile)
{
  Image3f image;
  if (filename.extension() == ".exr")
  {
    image = readEXR3(filename);
  }
  else if (filename.extension() == ".png")
  {
    image = readPNG3(filename);
    image /= 255.f; // convert from [0, 255] to [0, 1]
  }
  else
  {
    throw std::runtime_error("unsupported extension of " + filename.string());
  }
  
  // convert to linear space
  switch (colorSpaceInFile)
  {
  case ColorSpace::Linear: break;
  case ColorSpace::sRGB: clampAndPowInplace(image, 0.0f, std::numeric_limits<float>::max(), 2.2f); break;
  default: throw std::runtime_error("unsupported color space conversion");
  }
  
  std::unique_ptr<Texture<Vec3f> > result = std::make_unique<Texture<Vec3f> >(std::move(image));
  return result;
}

}
