#pragma once

#include <Hasty/VMath.h>

#include <robin_hood.h>

#include <unordered_map>
#include <array>

namespace Hasty
{

class CellIndex
{
public:
  CellIndex(int32_t x0, int32_t x1, int32_t x2) : x({ x0, x1, x2 }) {};

  std::array<int32_t, 3> x;
};

struct CellIndexHashF
{
  size_t operator()(const CellIndex& k) const
  {
    return robin_hood::hash_bytes(&k.x[0], k.x.size() * sizeof(int32_t));
  }
};

struct CellIndexEquals
{
  bool operator()(const CellIndex& lhs, const CellIndex& rhs) const
  {
    return lhs.x == rhs.x;
  }
};

typedef std::unordered_map<CellIndex, std::vector<uint32_t>, CellIndexHashF, CellIndexEquals> CellIndexMapStd;
typedef robin_hood::unordered_flat_map<CellIndex, std::vector<uint32_t>, CellIndexHashF, CellIndexEquals> CellIndexMapTSL;

class PointHashCells
{
  typedef CellIndexMapTSL HashMap;
public:

  void clear(float radius)
  {
    m_map.clear();
    m_points.clear();
    m_cellSize = radius;
  }

  CellIndex computeIndex(const Vec3f& p) const
  {
    return CellIndex(floor_int32(p[0] / m_cellSize), floor_int32(p[1] / m_cellSize), floor_int32(p[2] / m_cellSize));
  }
  void initialize(const std::vector<Vec3f>& points)
  {
    for (uint32_t idx = 0; idx < points.size(); idx++)
    {
      const Vec3f& p = points[idx];
      m_map[computeIndex(p)].push_back(idx);
    }
    m_points = points;
  }

  void printStats()
  {

  }

  void radiusNeighbors(const Vec3f& center, float radius, std::vector<std::uint32_t>& results) const
  {
    results.clear();
    float radiusSq = radius * radius;
    CellIndex pidxMin = computeIndex(center - Vec3f::Fill(radius));
    CellIndex pidxMax = computeIndex(center + Vec3f::Fill(radius));
    for (int32_t i = pidxMin.x[0]; i <= pidxMax.x[0]; i++)
    {
      for (int32_t j = pidxMin.x[1]; j <= pidxMax.x[1]; j++)
      {
        for (int32_t k = pidxMin.x[2]; k <= pidxMax.x[2]; k++)
        {
          auto iter = m_map.find(CellIndex(i, j, k));
          if (iter != m_map.end())
          {
            for (uint32_t pointIdx : (iter->second))
            {
              const Vec3f& x = m_points[pointIdx];
              if (normSq(x - center) <= radiusSq)
              {
                results.push_back(pointIdx);
              }
            }
          }
        }
      }
    }
  }

private:
  HashMap m_map;
  std::vector<Vec3f> m_points;
  float m_cellSize;
};

}
