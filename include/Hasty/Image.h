#pragma once

#include <Hasty/VMath.h>

#include <mutex>
#include <vector>
#include <cassert>
#include <string>
#include <cstdio>
#include <filesystem>

namespace Hasty
{

// a generic double buffer
// https://stackoverflow.com/questions/44888657/multithreading-shared-resource-periodically-copy-data-from-buffer-data-struc
template<typename _Data>
class DoubleBuffer
{

public:
  DoubleBuffer(unsigned int width, unsigned int height)
    :m_writeBuffer(width, height),
    m_readBuffer(width, height)
  {
  }

  void lock() { m_mutex.lock(); }
  bool try_lock() { return m_mutex.try_lock(); }
  void unlock() { m_mutex.unlock(); }

  auto& getWriteBuffer() { return m_writeBuffer; }
  auto& getReadBuffer() { return m_readBuffer; }

  void copyBuffer()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_readBuffer = m_writeBuffer;
  }

private:
  std::mutex m_mutex;

  _Data m_writeBuffer;
  _Data m_readBuffer;
};

enum class ColorSpace
{
  sRGB, Linear
};

template<class _PixelType>
class Image
{
public:
  typedef _PixelType PixelType;

  Image(){}
  Image(std::size_t width, std::size_t height)
  {
    resize(width, height);
  }
  std::size_t getWidth() const
  {
    return m_width;
  }
  std::size_t getHeight() const
  {
    return m_height;
  }
  void resize(std::size_t width, std::size_t height)
  {
    m_width = width;
    m_height = height;
    m_data.resize(m_width * m_height);
  }
  void setZero(std::size_t width, std::size_t height)
  {
    m_width = width;
    m_height = height;
    m_data.assign(m_width * m_height, PixelType::Zero());
  }
  void setOnes(std::size_t width, std::size_t height)
  {
    m_width = width;
    m_height = height;
    m_data.assign(m_width * m_height, PixelType::Ones());
  }
  void setConstant(std::size_t width, std::size_t height, const PixelType& v)
  {
    m_width = width;
    m_height = height;
    m_data.assign(m_width * m_height, v);
  }
  std::size_t size() const
  {
    return m_data.size();
  }

  PixelType& operator()(std::size_t x, std::size_t y)
  {
    PixelType& target_pixel = m_data[y * m_width + x];
    return target_pixel;
  }
  const PixelType& operator()(std::size_t x, std::size_t y) const
  {
    const PixelType& target_pixel = m_data[y * m_width + x];
    return target_pixel;
  }

  bool inside(int x, int y) const
  {
    return x >= 0 && y >= 0 && x < m_width&& y < m_height;
  }

  Image& operator+=(const Image& image2)
  {
    assert(m_width == image2.m_width);
    assert(m_height == image2.m_height);
    for (int i = 0; i < m_width * m_height; i++)
    {
      m_data[i] += image2.m_data[i];
    }
    return *this;
  }
  Image& operator*=(float s)
  {
    for (int i = 0; i < m_width * m_height; i++)
    {
      m_data[i] *= s;
    }
    return *this;
  }
  Image& operator/=(float s)
  {
    for (int i = 0; i < m_width * m_height; i++)
    {
      m_data[i] /= s;
    }
    return *this;
  }

  PixelType* data()
  {
    return m_data.data();
  }
  const PixelType* data() const
  {
    return m_data.data();
  }


protected:
  std::vector<PixelType> m_data;
  std::size_t m_width, m_height;

};


struct templateDummyType {};

template<typename T>
inline T zero()
{
  static_assert(!std::is_same<T, templateDummyType>::value, "this function is not supposed to work, see other specializations");
}

template <>
inline float zero()
{
  return 0.0f;
}

template<int W, int H, typename _PixelType>
Image<_PixelType> runSmallFilter(const Image<_PixelType> &image, const std::array<float, W * H>& kernel)
{
  static_assert(W % 2 == 1);
  static_assert(H % 2 == 1);
  std::size_t width = image.getWidth();
  std::size_t height = image.getHeight();
  assert(W < width);
  assert(H < height);
  Image<_PixelType> result(width, height);
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      _PixelType pi = zero<_PixelType>();
      for (int j = 0; j < H; j++)
      {
        for (int i = 0; i < W; i++)
        {
          int i2 = x + i - W / 2;
          int j2 = y + j - H / 2;
          // apply "boundary" condition
          int i3 = i2 < 0 ? 0 : ( i2 >= width ? width - 1 : i2 );
          int j3 = j2 < 0 ? 0 : ( j2 >= height ? height - 1 : j2 );
          pi += kernel[j * W + i] * image(i3, j3);
        }
      }
      result(x, y) = pi;
    }
  }
  return result;
}

template<typename _PixelTypeOut, typename _PixelTypeIn, typename Functor>
Image<_PixelTypeOut> transform(const Image<_PixelTypeIn> &image, Functor f)
{
  std::size_t width = image.getWidth();
  std::size_t height = image.getHeight();
  Image<_PixelTypeOut> result(width, height);
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      result(x, y) = f(x, y, image(x, y));
    }
  }
  return result;
}

template<typename _PixelTypeIn>
const _PixelTypeIn* begin(const Image<_PixelTypeIn> &image)
{
  const _PixelTypeIn* b = image.data();
  return b;
}
template<typename _PixelTypeIn>
const _PixelTypeIn* end(const Image<_PixelTypeIn> &image)
{
  const _PixelTypeIn* b = image.data();
  const _PixelTypeIn* e = b + image.getHeight() * image.getWidth();
  return e;
}
template<typename _PixelTypeIn>
_PixelTypeIn* begin(Image<_PixelTypeIn> &image)
{
  _PixelTypeIn* b = image.data();
  return b;
}
template<typename _PixelTypeIn>
_PixelTypeIn* end(Image<_PixelTypeIn> &image)
{
  _PixelTypeIn* b = image.data();
  _PixelTypeIn* e = b + image.getHeight() * image.getWidth();
  return e;
}

typedef Image<float> Image1f;
typedef Image<Vec3f> Image3f;
typedef Image<Vec4f> Image4f;

Image3f get3channelsFlipUpDown(const Image4f& image);
Image3f flipUpDown(const Image3f& image);
void writePFM(const Image1f& image, const std::filesystem::path& filename);
void writePFM(const Image3f& image, const std::filesystem::path& filename);
Image3f readEXR3(const std::filesystem::path& filename);
void writeEXR(const Image<Vec3f>& image, const std::filesystem::path& filename);
Image1f readPNG1(const std::filesystem::path& filename);
Image3f readPNG3(const std::filesystem::path& filename);

template<typename C>
void swap(Image<C>& i1, Image<C>& i2)
{
  std::swap(i1.m_data, i2.m_data);
  std::swap(i1.m_width, i2.m_width);
  std::swap(i1.m_height, i2.m_height);
}

template<class _PixelType>
class AccumulationBuffer
{
public:
  Image<_PixelType> data;
  int numSamples;

  AccumulationBuffer(unsigned int width, unsigned int height)
  {
    data.setZero(width, height);
    numSamples = 0;
  }
};

typedef DoubleBuffer<AccumulationBuffer<Vec3f> > Image3fAccDoubleBuffer;
typedef DoubleBuffer<AccumulationBuffer<Vec4f> > Image4fAccDoubleBuffer;


template<typename PixelType>
void clampAndPowInplace(Image<PixelType> & image, float low, float high, float exponent)
{
  std::size_t width = image.getWidth();
  std::size_t height = image.getHeight();

  for (std::size_t i = 0; i < width; i++)
  {
    for (std::size_t j = 0; j < height; j++)
    {
      image(i, j) = pow(clamp(image(i, j), low, high), exponent);
    }
  }
}

}
