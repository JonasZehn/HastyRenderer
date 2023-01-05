#include <Hasty/Random.h>

extern "C"
{
#include <shishua.h>
}

namespace Hasty
{

CopyablePrng_state::CopyablePrng_state(uint64_t seed)
{
  unalignedPrngState = makeUniqueAlignmentBuffer<prng_state>();

  uint64_t seed_zero[4] = { seed, 0, 0, 0 };
  prng_init(getState(), seed_zero);
  offset = bufferSize;
}

CopyablePrng_state::CopyablePrng_state(const CopyablePrng_state& s2)
{
  unalignedPrngState = makeUniqueAlignmentBuffer<prng_state>();
  this->operator=(s2);
}
CopyablePrng_state::~CopyablePrng_state()
{
}
CopyablePrng_state& CopyablePrng_state::operator=(const CopyablePrng_state& s2)
{
  memcpy(getState(), s2.getState(), sizeof(prng_state));
  this->offset = s2.offset;
  this->buf = s2.buf;
  return *this;
}
prng_state* CopyablePrng_state::getState()
{
  return getAutoAligned<prng_state>(unalignedPrngState.get());
}
prng_state const* CopyablePrng_state::getState() const
{
  return getAutoAligned<const prng_state>(unalignedPrngState.get());
}
void CopyablePrng_state::generateMoreData()
{
  prng_gen(getState(), (uint8_t*)buf.data(), sizeof(uint32_t) * bufferSize);
  offset = 0;
}

}
