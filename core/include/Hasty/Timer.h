#pragma once

#include <chrono>

namespace Hasty
{

class HighResTimer
{
public:
  HighResTimer()
  {
    m_start = std::chrono::high_resolution_clock::now();
  }
  double seconds() const
  {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(stop - m_start);
    return duration.count();
  }
private:
  std::chrono::high_resolution_clock::time_point m_start;
};


}
