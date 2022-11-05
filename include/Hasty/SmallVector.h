#pragma once

#include <vector>
#include <array>

namespace Hasty
{

template<typename T, std::size_t N>
class SmallVector
{
public:
  SmallVector()
    :m_size(0)
  {
  }
  SmallVector(int size)
    :m_size(0)
  {
    this->resize(size);
  }
  void emplace_back(T&& v)
  {
    if (m_size >= N)
    {
      m_v.emplace_back(std::move(v));
    }
    else
    {
      m_a[m_size] = v;
    }
    m_size += 1;
  }
  std::size_t size() const
  {
    return m_size;
  }
  void resize(std::size_t newsize)
  {
    if (newsize < N) m_v.clear();
    else m_v.resize(newsize - N);

    m_size = newsize;
  }
  const T& operator[](std::size_t index) const
  {
    if (index >= N)
    {
      return m_v[index - N];
    }
    else
    {
      return m_a[index];
    }
  }
  T& operator[](std::size_t index)
  {
    return const_cast<T&>(const_cast<const SmallVector<T, N> *>(this)->operator[](index));
  }
  void remove(std::size_t index)
  {
    assert(index < m_size&& m_size > 0);
    for (std::size_t j = index; j < m_size - 1; j++)
    {
      this->operator[](j) = this->operator[](j + 1);
    }
    m_size -= 1;
  }

private:
  std::array<T, N> m_a;
  std::vector<T> m_v;
  std::size_t m_size;
};

}
