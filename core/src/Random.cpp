#include <Hasty/Random.h>

#ifdef use_shishua

extern "C"
{
#include <shishua.h>
}

namespace Hasty
{

CopyablePrng_state::CopyablePrng_state()
{
  m_data = std::make_unique<prng_state>();
}

CopyablePrng_state::CopyablePrng_state(const CopyablePrng_state& s2)
{
  m_data = std::make_unique<prng_state>();
  memcpy(m_data.get(), s2.m_data.get(), sizeof(prng_state));
}
CopyablePrng_state::~CopyablePrng_state()
{
}
CopyablePrng_state& CopyablePrng_state::operator=(const CopyablePrng_state& s2)
{
  memcpy(m_data.get(), s2.m_data.get(), sizeof(prng_state));
  return *this;
}

RNG::RNG(uint64_t seed)
{
  uint64_t seed_zero[4] = { seed, 0, 0, 0 };
  prng_init(state.get(), seed_zero);
  offset = bufferSize;
}
RNG::~RNG() {}
uint32_t RNG::get4bytes()
{
  if(offset >= bufferSize)
  {
    prng_gen(state.get(), (uint8_t*)buf.data(), sizeof(uint32_t) * bufferSize);
    offset = 0;
  }
  uint32_t result = buf[offset];
  offset += 1;
  return result;
}

}


#endif
