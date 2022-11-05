#pragma once

#include <Hasty/VMath.h>

#include <random>
#include <memory>

#define use_shishua

namespace Hasty
{

class HaltonSequence
{
  typedef uint32_t NumberType;
  NumberType b;
  NumberType n;
  NumberType d;
public:
  HaltonSequence(NumberType b)
    :b(b)
  {
    n = 0;
    d = 1;
  }

  float operator()()
  {
    NumberType x = d - n;
    if (x == 1)
    {
      n = 1;
      d *= b;
    }
    else
    {
      NumberType y = d / b;
      while (x <= y)
      {
        y /= b;
      }
      n = (b + 1) * y - x;
    }
    return float(n) / d;
  }
};

class HaltonSequence2
{
  HaltonSequence s1;
  HaltonSequence s2;
public:
  HaltonSequence2()
    :s1(2), s2(3)
  {

  }
  Vec2f operator()()
  {
    return Vec2f(s1(), s2());
  }
};

}

#ifndef use_shishua

namespace Hasty
{

// we implement https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator
// so it can be passed to discrete distribution
class RNG
{
public:

  std::default_random_engine rengine;
  std::uniform_real_distribution<float> dist;
  typedef std::default_random_engine::result_type result_type;

  RNG(uint64_t seed)
    : dist(0.0f, 1.0f)
  {
    rengine.seed(seed);
  }

  float uniform01()
  {
    return dist(rengine);
  }

  float uniformm11()
  {
    return 2.0f * uniform01() - 1.0f;
  }

  result_type min() const
  {
    return rengine.min();
  }
  result_type max() const
  {
    return rengine.max();
  }

  result_type operator()()
  {
    return rengine();
  }
};

}

#else

struct prng_state;

namespace Hasty
{

class CopyablePrng_state  // want to make this copyable so we can more easily restore and debug state while decoupling dependency on prng
{
public:
  CopyablePrng_state();
  CopyablePrng_state(const CopyablePrng_state& s2);
  ~CopyablePrng_state();
  CopyablePrng_state& operator=(const CopyablePrng_state& s2);


  prng_state* get() { return m_data.get(); }
private:
  std::unique_ptr<prng_state> m_data;
};

class RNG
{
public:

  typedef uint32_t result_type;

  static const size_t bufferSize = 128;

  RNG(uint64_t seed);
  ~RNG();

  uint32_t get4bytes();
  
  /** returns a value in the range [0, 1) */
  float uniform01f()
  {
    uint32_t u32 = get4bytes();
    float result = float(u32 & uint32_t(0x007FFFFF) ) * std::powf(2, -23.f); // this is faster than std::ldexpf(u32, -23), thats so wierd
    assert(result < 1.0f);
    return result;
  }
  
  Vec2f uniform01Vec2f()
  {
    return Vec2f(uniform01f(), uniform01f());
  }

  Vec3f uniform01Vec3f()
  {
    return Vec3f(uniform01f(), uniform01f(), uniform01f());
  }

  float uniformm11f()
  {
    return 2.0f * uniform01f() - 1.0f;
  }

  result_type min() const
  {
    return std::numeric_limits<uint32_t>::min();
  }
  result_type max() const
  {
    return std::numeric_limits<uint32_t>::max();
  }

  result_type operator()()
  {
    uint32_t u32 = get4bytes();
    return u32;
  }

private:
  CopyablePrng_state state;
  std::array<uint32_t, bufferSize> buf = { 0 };
  uint32_t offset;
};

}
#endif // #ifndef use_shishua
