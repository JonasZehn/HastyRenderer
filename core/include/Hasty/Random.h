#pragma once

#include <Hasty/VMath.h>

#include <random>
#include <memory>
#include <array>

struct prng_state;

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
    if(x == 1)
    {
      n = 1;
      d *= b;
    }
    else
    {
      NumberType y = d / b;
      while(x <= y)
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

template<typename T>
T* getAutoAligned(void* ptr)
{
  constexpr std::size_t alignment = alignof(T);
  std::size_t stateSize = sizeof(T) + alignment - 1;
  T* result = reinterpret_cast<T*>(std::align(alignment, sizeof(prng_state), ptr, stateSize));
  assert(result != nullptr);
  return result;
}

using UniqueAlignmentBuffer = std::unique_ptr<uint8_t[]>;

template<typename T>
UniqueAlignmentBuffer makeUniqueAlignmentBuffer()
{
  constexpr std::size_t alignment = alignof(T);
  std::size_t stateSize = sizeof(T) + alignment - 1;
  return std::unique_ptr<uint8_t[]>(new uint8_t[stateSize]);
}

class CopyablePrng_state  // want to make this copyable so we can more easily restore and debug state while decoupling dependency on prng
{
public:
  CopyablePrng_state(uint64_t seed);
  CopyablePrng_state(const CopyablePrng_state& s2);
  CopyablePrng_state(CopyablePrng_state&& s2) = delete;
  ~CopyablePrng_state();
  CopyablePrng_state& operator=(const CopyablePrng_state& s2);
  CopyablePrng_state& operator=(CopyablePrng_state&& s2) = delete;

  inline uint32_t CopyablePrng_state::next4Bytes()
  {
    if(offset >= bufferSize)
    {
      generateMoreData(); // this way the hot path can be inlined, without including shishua in this header file and poluting the namespace
    }
    uint32_t result = buf[offset];
    offset += 1;
    return result;
  }

private:
  prng_state* getState();
  prng_state const* getState() const;
  void generateMoreData();

  UniqueAlignmentBuffer unalignedPrngState;

  static const size_t bufferSize = 128;
  std::array<uint32_t, bufferSize> buf = { 0 };
  uint32_t offset;
};

class RNG
{
public:

  typedef uint32_t result_type;

  inline RNG(uint64_t seed)
    :state(seed)
  {
  }
  inline ~RNG() {}

  /** returns a value in the range [0, 1) */
  inline float uniform01f()
  {
    uint32_t u32 = state.next4Bytes();
    float result = float(u32 & uint32_t(0x007FFFFF)) * std::pow(2.0f, -23.f); // this is faster than std::ldexpf(u32, -23), thats so wierd
    assert(result < 1.0f);
    return result;
  }

  inline float uniformm11f()
  {
    return 2.0f * uniform01f() - 1.0f;
  }

  inline result_type min() const
  {
    return std::numeric_limits<uint32_t>::min();
  }
  inline result_type max() const
  {
    return std::numeric_limits<uint32_t>::max();
  }

  inline result_type operator()()
  {
    uint32_t u32 = state.next4Bytes();
    return u32;
  }
  inline uint32_t next4Bytes()
  {
    return state.next4Bytes();
  }

private:
  CopyablePrng_state state;
};


inline float uniform01f(RNG& rng)
{
  return rng.uniform01f();
}

}
