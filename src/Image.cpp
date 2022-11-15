#include <Hasty/Image.h>

#include <OpenImageIO/imageio.h>

namespace Hasty
{


Image3f get3channelsFlipUpDown(const Image4f& image)
{
  Image3f subImage;
  subImage.resize(image.getWidth(), image.getHeight());
  for (int i = 0; i < image.getHeight(); i++)
  {
    for (int j = 0; j < image.getWidth(); j++)
    {
      subImage(j, image.getHeight() - i - 1) = Vec3f(image(j, i)[0], image(j, i)[1], image(j, i)[2]);
    }
  }
  return subImage;
}

Image3f flipUpDown(const Image3f& image)
{
  Image3f subImage;
  subImage.resize(image.getWidth(), image.getHeight());
  for (int i = 0; i < image.getHeight(); i++)
  {
    for (int j = 0; j < image.getWidth(); j++)
    {
      subImage(j, image.getHeight() - i - 1) = image(j, i);
    }
  }
  return subImage;
}


bool isLittleEndian()
{
  uint32_t value = 0x03020100; // 0x03020100
  uint8_t* endianValues = (uint8_t*)&value;
  return endianValues[0] == 0x00 && endianValues[1] == 0x01 && endianValues[2] == 0x02 && endianValues[3] == 0x03;
}

void writePFM(const Image1f& image, const std::filesystem::path& filename)
{
  // https://www.pauldebevec.com/Research/HDR/PFM/

  FILE* filestream = fopen(filename.string().c_str(), "wb");
  if (filestream == NULL)
  {
    throw std::runtime_error("could not open file " + filename.string() + "for writing!");
  }

  fputs("Pf\n", filestream);
  fprintf(filestream, "%d %d\n", (int)image.getWidth(), (int)image.getHeight());

  if (!isLittleEndian())
  {
    throw std::runtime_error(filename.string() + " only supporting little endian export ");
  }

  fputs("-1.0\n", filestream);
  fwrite(image.data(), sizeof(float) * image.getWidth() * image.getHeight(), 1, filestream);

  fclose(filestream);
}
void writePFM(const Image3f& image, const std::filesystem::path& filename)
{
  FILE* filestream = fopen(filename.string().c_str(), "wb");
  if (filestream == NULL)
  {
    throw std::runtime_error("could not open file " + filename.string() + "for writing");
  }

  fputs("PF\n", filestream);
  fprintf(filestream, "%d %d\n", (int)image.getWidth(), (int)image.getHeight());


  if (!isLittleEndian())
  {
    throw std::runtime_error(filename.string() + " only supporting little endian export ");
  }

  fputs("-1.0\n", filestream);
  fwrite(image.data(), sizeof(float) * 3 * image.getWidth() * image.getHeight(), 1, filestream);

  fclose(filestream);
}
Image1f readImage1f(const std::filesystem::path& filename)
{
  auto inp = OIIO::ImageInput::open(filename);
  if (!inp)
  {
    throw std::runtime_error("could not open file " + filename.string() + " for reading");
  }
  const OIIO::ImageSpec &spec = inp->spec();
  int width = spec.width;
  int height = spec.height;
  int numChannels = spec.nchannels;
  if (numChannels != 1)
  {
    throw std::runtime_error("could not read " + filename.string() + ", wrong number of channels, expected 1, found " + std::to_string(numChannels));
  }
  
  Image1f image;
  image.resize(width, height);
  bool success = inp->read_image(OIIO::TypeDesc::FLOAT, image.data());
  if (!success)
  {
    throw std::runtime_error("failed to read " + filename.string() );
  }
  
  inp->close();

  return image;
}
Image3f readImage3f(const std::filesystem::path& filename)
{
  auto inp = OIIO::ImageInput::open(filename);
  if (!inp)
  {
    throw std::runtime_error("could not open file " + filename.string() + " for reading");
  }
  const OIIO::ImageSpec &spec = inp->spec();
  int width = spec.width;
  int height = spec.height;
  int numChannels = spec.nchannels;
  if (numChannels != 3 && numChannels != 4)
  {
    throw std::runtime_error("could not read " + filename.string() + ", wrong number of channels, expected 3 or 4, found " + std::to_string(numChannels));
  }

  std::vector<float> pixels(width * height * numChannels);
  bool success = inp->read_image(OIIO::TypeDesc::FLOAT, pixels.data());
  if (!success)
  {
    throw std::runtime_error("failed to read " + filename.string() );
  }
  
  Image3f image;
  image.resize(width, height);
  inp->close();
  
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      std::size_t offset = width * numChannels * y + numChannels * x;
      image(x, y) = Vec3f(pixels[offset + 0], pixels[offset + 1], pixels[offset + 2]);
    }
  }

  return image;
}
void writeEXR(const Image<Vec3f>& image, const std::filesystem::path& filename)
{
  std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create (filename);
  if (!out)
  {
    throw std::runtime_error("could not open file " + filename.string() + " for writing");
  }
  OIIO::TypeDesc typeDesc = OIIO::TypeDesc::FLOAT;
  OIIO::ImageSpec spec(image.getWidth(), image.getHeight(), 3, typeDesc);
  out->open (filename, spec);
  out->write_image(typeDesc, image.data());
  out->close ();
}

}
