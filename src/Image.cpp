#include <Hasty/Image.h>

#include <ImfHeader.h>
#include <ImfRgbaFile.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfChannelList.h>
#include <ImfFramebuffer.h>
#include <ImfBoxAttribute.h>


#include <png.h>

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
Image3f readEXR3(const std::filesystem::path& filename)
{
  using namespace Imf_3_1;
  using namespace Imath;

  Image<Vec3f> image;

  InputFile file (filename.string().c_str());
  Box2i dw = file.header().dataWindow();
  uint32_t width = dw.max.x - dw.min.x + 1;
  uint32_t height = dw.max.y - dw.min.y + 1;
  image.resize(width, height);
  
  Imf_3_1::FrameBuffer frameBuffer;
  frameBuffer.insert("R", Slice(FLOAT, (char*)image.data(), sizeof(Vec3f), sizeof(Vec3f) * image.getWidth()));
  frameBuffer.insert("G", Slice(FLOAT, ((char*)image.data()) + sizeof(float) * 1, sizeof(Vec3f), sizeof(Vec3f) * image.getWidth()));
  frameBuffer.insert("B", Slice(FLOAT, ((char*)image.data()) + sizeof(float) * 2, sizeof(Vec3f), sizeof(Vec3f) * image.getWidth()));

  file.setFrameBuffer (frameBuffer);
  file.readPixels (dw.min.y, dw.max.y);
  return image;
}
void writeEXR(const Image<Vec3f>& image, const std::filesystem::path& filename)
{
  using namespace Imf_3_1;

  Header header(image.getWidth(), image.getHeight());
  header.channels().insert("R", Channel(FLOAT));
  header.channels().insert("G", Channel(FLOAT));
  header.channels().insert("B", Channel(FLOAT));
  Imf_3_1::OutputFile file(filename.string().c_str(), header);
  Imf_3_1::FrameBuffer frameBuffer;
  frameBuffer.insert("R", Slice(FLOAT, (char*)image.data(), sizeof(Vec3f), sizeof(Vec3f) * image.getWidth()));
  frameBuffer.insert("G", Slice(FLOAT, ((char*)image.data()) + sizeof(float) * 1, sizeof(Vec3f), sizeof(Vec3f) * image.getWidth()));
  frameBuffer.insert("B", Slice(FLOAT, ((char*)image.data()) + sizeof(float) * 2, sizeof(Vec3f), sizeof(Vec3f) * image.getWidth()));
  file.setFrameBuffer(frameBuffer);
  file.writePixels(image.getHeight());
}
Image1f readPNG1(const std::filesystem::path& filename)
{
  FILE *fp = fopen(filename.string().c_str(), "rb");
  if (fp == nullptr)
  {
    throw std::runtime_error(" could not open " + filename.string() + " for reading ");
  }
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);  // <-- creating a new, local info_ptr 
  png_init_io(png_ptr, fp);
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_STRIP_ALPHA | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, NULL);
  uint32_t width = png_get_image_width(png_ptr, info_ptr);
  uint32_t height = png_get_image_height(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  if (color_type != PNG_COLOR_TYPE_GRAY)
  {
    throw std::runtime_error(filename.string() + " not supported texture format ");
  }
  png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  if (bit_depth != 8)
  {
    throw std::runtime_error(filename.string() + " not supported texture format; bit depth ");
  }
  png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
  std::size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
  
  int numChannels = 1;

  assert(row_bytes == numChannels * width);
  std::unique_ptr<uint8_t[]> outData = std::make_unique<uint8_t[]>(row_bytes * height);
  for (uint32_t i = 0; i < height; i++) {
      memcpy(outData.get() + (row_bytes * i), row_pointers[i], row_bytes);
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

  fclose(fp);

  
  Image1f image;
  image.resize(width, height);

  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      std::size_t index = width * numChannels * y + numChannels * x;
      unsigned char c = outData[index];
      image(x, y) = float(c);
    }
  }
  return image;
}
Image3f readPNG3(const std::filesystem::path& filename)
{
  FILE *fp = fopen(filename.string().c_str(), "rb");
  if (fp == nullptr)
  {
    throw std::runtime_error(" could not open " + filename.string() + " for reading ");
  }
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);  // <-- creating a new, local info_ptr 
  png_init_io(png_ptr, fp);
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_STRIP_ALPHA | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, NULL);
  uint32_t width = png_get_image_width(png_ptr, info_ptr);
  uint32_t height = png_get_image_height(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  if (color_type != PNG_COLOR_TYPE_RGB)
  {
    throw std::runtime_error(filename.string() + " not supported texture format ");
  }
  png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  if (bit_depth != 8)
  {
    throw std::runtime_error(filename.string() + " not supported texture format; bit depth ");
  }
  png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
  std::size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
  
  int numChannels = 3;

  assert(row_bytes == numChannels * width);
  std::unique_ptr<uint8_t[]> outData = std::make_unique<uint8_t[]>(row_bytes * height);
  for (uint32_t i = 0; i < height; i++) {
      memcpy(outData.get() + (row_bytes * i), row_pointers[i], row_bytes);
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

  fclose(fp);

  
  Image3f image;
  image.resize(width, height);

  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      for (std::size_t k = 0; k < numChannels; k++)
      {
        std::size_t index = width * numChannels * y + numChannels * x + k;
        unsigned char c = outData[index];
        image(x, y)[k] = float(c);
      }
    }
  }
  return image;
}


}