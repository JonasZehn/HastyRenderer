#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdint>

namespace Hasty
{

extern const double Pi;
extern const double InvPi;
inline float square(float f) { return f * f; }

inline int32_t floor_int32(double x)
{
  int i = (int)x; // remove digits after decimal point
  return i - (i > x); // need to subtract 1 when negative
}
// computes fmod(fmod(f, 1) + 1, 1)
inline float fmod1p1(float f)
{
  return f - Hasty::floor_int32(f);
}

template<int N>
constexpr float powci(float f)
{
  static_assert(N >= 0, "only support non negative exponents, use powf instead");
  if constexpr (N == 0) return 1.0f;
  else
  {
    float q = powci<N / 2>(f);
    if constexpr (N % 2 == 1)
    {
      return f * (q * q);
    }
    else
    {
      return q * q;
    }
  }
}



class Vec2f
{
public:
  Vec2f()
  {
    m_data[0] = std::numeric_limits<float>::signaling_NaN();
    m_data[1] = std::numeric_limits<float>::signaling_NaN();
  }
  Vec2f(float x, float y)
  {
    m_data[0] = x;
    m_data[1] = y;
  }
  float& operator[](std::size_t i)
  {
    return m_data[i];
  }
  const float& operator[](std::size_t i) const
  {
    return m_data[i];
  }
  Vec2f operator+(const Vec2f& v2) const
  {
    return Vec2f(m_data[0] + v2[0], m_data[1] + v2[1]);
  }
  Vec2f operator*(float f) const
  {
    return Vec2f(m_data[0] * f, m_data[1] * f);
  }
  Vec2f& operator+=(const Vec2f& v2)
  {
    m_data[0] += v2[0];
    m_data[1] += v2[1];
    return *this;
  }
  static Vec2f Constant(float f)
  {
    return Vec2f(f, f);
  }
  static Vec2f Zero()
  {
    return Constant(0.0f);
  }
  static Vec2f Ones()
  {
    return Constant(1.0f);
  }

private:
  std::array<float, 2> m_data;
};

class Vec3f
{
public:
  typedef float ScalarType;

  Vec3f()
  {
    m_data[0] = std::numeric_limits<float>::signaling_NaN();
    m_data[1] = std::numeric_limits<float>::signaling_NaN();
    m_data[2] = std::numeric_limits<float>::signaling_NaN();
  }
  Vec3f(float x, float y, float z)
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
  }
  float& operator[](std::size_t i)
  {
    return m_data[i];
  }
  const float& operator[](std::size_t i) const
  {
    return m_data[i];
  }
  Vec3f operator+(const Vec3f& v2) const
  {
    return Vec3f(m_data[0] + v2[0], m_data[1] + v2[1], m_data[2] + v2[2]);
  }
  Vec3f& operator+=(const Vec3f& v2)
  {
    m_data[0] += v2[0];
    m_data[1] += v2[1];
    m_data[2] += v2[2];
    return *this;
  }
  Vec3f operator/(float s) const
  {
    return Vec3f(m_data[0] / s, m_data[1] / s, m_data[2] / s);
  }
  Vec3f& operator*=(float s)
  {
    m_data[0] *= s;
    m_data[1] *= s;
    m_data[2] *= s;
    return *this;
  }
  Vec3f& operator/=(float s)
  {
    m_data[0] /= s;
    m_data[1] /= s;
    m_data[2] /= s;
    return *this;
  }
  Vec3f operator-() const
  {
    return Vec3f(-m_data[0], -m_data[1], -m_data[2]);
  }
  Vec3f operator-(const Vec3f& v2) const
  {
    return Vec3f(m_data[0] - v2[0], m_data[1] - v2[1], m_data[2] - v2[2]);
  }
  Vec3f operator*(float s) const
  {
    return Vec3f(m_data[0]*s, m_data[1] *s, m_data[2] *s);
  }
  bool operator==(const Vec3f& v2) const
  {
    return m_data[0] == v2[0] && m_data[1] == v2[1] && m_data[2] == v2[2];
  }
  bool operator!=(const Vec3f& v2) const
  {
    return !this->operator==(v2);
  }
  Vec3f cross(const Vec3f& v2) const
  {
    return Vec3f(
      m_data[1] * v2[2] - m_data[2] * v2[1],
      m_data[2] * v2[0] - m_data[0] * v2[2],
      m_data[0] * v2[1] - m_data[1] * v2[0]
    );
  }
  float dot(const Vec3f& v2) const
  {
    return m_data[0] * v2[0] + m_data[1] * v2[1] + m_data[2] * v2[2];
  }
  float normSq() const
  {
    return this->dot(*this);
  }
  float norm() const
  {
    return std::sqrtf(this->dot(*this));
  }
  float normL1() const
  {
    return cwiseAbs().sum();
  }
  float sum() const
  {
    return m_data[0] + m_data[1] + m_data[2];
  }
  void normalize()
  {
    (*this) /= this->norm();
  }
  Vec3f normalized() const
  {
    return (*this) / this->norm();
  }
  Vec3f cwiseAbs() const
  {
    return Vec3f(std::abs(m_data[0]), std::abs(m_data[1]), std::abs(m_data[2]));
  }
  Vec3f cwiseProd(const Vec3f& v2) const
  {
    return Vec3f(m_data[0] * v2[0], m_data[1] * v2[1], m_data[2] * v2[2]);
  }
  Vec3f cwiseExp() const
  {
    return Vec3f(std::exp(m_data[0]), std::exp(m_data[1]), std::exp(m_data[2]));
  }
  Vec3f cwiseLog() const
  {
    return Vec3f(std::log(m_data[0]), std::log(m_data[1]), std::log(m_data[2]));
  }
  Vec3f cwiseSqrt() const
  {
    return Vec3f(std::sqrtf(m_data[0]), std::sqrtf(m_data[1]), std::sqrtf(m_data[2]));
  }
  Vec3f cwisePow(float exponent) const
  {
    return Vec3f(std::pow(m_data[0], exponent), std::pow(m_data[1], exponent), std::pow(m_data[2], exponent));
  }
  bool isFinite() const
  {
    return std::isfinite(m_data[0]) && std::isfinite(m_data[1]) && std::isfinite(m_data[2]);
  }
  
  Vec3f clamp(float bottom, float top) const
  {
    return Vec3f(
      std::max(bottom, std::min(top, m_data[0])),
      std::max(bottom, std::min(top, m_data[1])),
      std::max(bottom, std::min(top, m_data[2]))

    );
  }

  static Vec3f Constant(float f)
  {
    return Vec3f(f, f, f);
  }
  static Vec3f Zero()
  {
    return Constant(0.0f);
  }
  static Vec3f Ones()
  {
    return Constant(1.0f);
  }

  std::size_t size() const
  {
    return m_data.size();
  }

private:
  std::array<float, 3> m_data;
};

inline float mix(float v1, float v2, float lambda)
{
  return (1.0f - lambda) * v1 + lambda * v2;
}
inline Vec3f mix(const Vec3f& v1, const Vec3f& v2, float lambda)
{
  Vec3f result;
  for (int i = 0; i < result.size(); i++)
  {
    result[i] = (1.0f - lambda) * v1[i] + lambda * v2[i];
  }
  return result;
}

inline Vec3f operator*(float s, const Vec3f& v)
{
  return v * s;
}

inline std::ostream& operator<<(std::ostream& out, const Vec3f& v)
{
  out << v[0] << ' ' << v[1] << ' ' << v[2];
  return out;
}

void from_json(const nlohmann::json& j, Vec3f& v);


class Vec4f
{
public:
  Vec4f()
  {
    m_data[0] = std::numeric_limits<float>::signaling_NaN();
    m_data[1] = std::numeric_limits<float>::signaling_NaN();
    m_data[2] = std::numeric_limits<float>::signaling_NaN();
    m_data[3] = std::numeric_limits<float>::signaling_NaN();
  }
  Vec4f(float x, float y, float z, float w)
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = w;
  }
  Vec4f& operator+=(const Vec4f& v2)
  {
    m_data[0] += v2[0];
    m_data[1] += v2[1];
    m_data[2] += v2[2];
    m_data[3] += v2[3];
    return *this;
  }
  Vec4f operator/(float s) const
  {
    return Vec4f(m_data[0] / s, m_data[1] / s, m_data[2] / s, m_data[3] / s);
  }
  Vec4f& operator/=(float s)
  {
    m_data[0] /= s;
    m_data[1] /= s;
    m_data[2] /= s;
    m_data[3] /= s;
    return *this;
  }
  float& operator[](std::size_t i)
  {
    return m_data[i];
  }
  const float& operator[](std::size_t i) const
  {
    return m_data[i];
  }
  Vec4f clamp(float bottom, float top) const
  {
    return Vec4f(
      std::max(bottom, std::min(top, m_data[0])),
      std::max(bottom, std::min(top, m_data[1])),
      std::max(bottom, std::min(top, m_data[2])),
      std::max(bottom, std::min(top, m_data[3]))

    );
  }
  
  static Vec4f Constant(float f)
  {
    return Vec4f(f, f, f, f);
  }
  static Vec4f Zero()
  {
    return Constant(0.0f);
  }
  static Vec4f Ones()
  {
    return Constant(1.0f);
  }


private:
  std::array<float, 4> m_data;
};

inline void assertUnitLength(const Vec3f &v)
{
  assert(std::abs(v.norm() - 1) < 1e-4f);
}

inline void assertFinite(const Vec3f &v)
{
  assert(v.isFinite());
}
inline void assertFinite(float v)
{
  assert(std::isfinite(v));
}

template<class ScalarType, class VectorType>
class RodriguesRotation
{
public:

  RodriguesRotation(const VectorType& vFrom, const VectorType& vTo)
  {
    assertUnitLength(vFrom);
    assertUnitLength(vTo);
    cosAngle = vFrom.dot(vTo);
    kSinAngle = vFrom.cross(vTo);
  }
  
  VectorType operator*(const VectorType& x) const
  {
    if (std::abs(cosAngle - ScalarType(-1) ) > std::numeric_limits<ScalarType>::epsilon())
    {
      return x * cosAngle + kSinAngle.cross(x) + kSinAngle * (kSinAngle.dot(x) / (ScalarType(1.0) + cosAngle));
    }
    else
    {
      return -x;
    }
  }
  VectorType applyInverse(const VectorType& x) const
  {
    // we can just exchange vfrom and vto
    // the dot product is the same in the opposite direction, and opposite cross product is just the negative, so we use the negative of k
    if (std::abs(cosAngle - ScalarType(-1) ) > std::numeric_limits<ScalarType>::epsilon())
    {
      return x * cosAngle - kSinAngle.cross(x) + kSinAngle * (kSinAngle.dot(x) / (ScalarType(1.0) + cosAngle));
    }
    else
    {
      return -x;
    }
  }

private:
  ScalarType cosAngle;
  VectorType kSinAngle;
};

class Ray
{
public:
  Ray(){}
  Ray(const Vec3f& origin, const Vec3f& direction)
  {
    m_origin = origin;
    m_direction = direction;
  }
  const Vec3f& origin() const
  {
    return m_origin;
  }
  const Vec3f& direction() const
  {
    return m_direction;
  }

private:
  Vec3f m_origin;
  Vec3f m_direction;

};


/**
* outg    N   return value
*    |\` /|\ `/|
*       \ | /
* ------------------
*          \
*           _\|
*             -outg
*/
inline Vec3f reflectAcross(const Vec3f& outgoing, const Vec3f& N)
{
  return  -outgoing + 2.0f * N.dot(outgoing) * N;
}
/**
* outg    N   
*    |\` /|\
*       \ |
* ------------------
*       /
*     |/_
*   return value
*/
inline Vec3f reflectAcrossNormalPlane(const Vec3f& outgoing, const Vec3f& N)
{
  return  outgoing - 2.0f * N.dot(outgoing) * N;
}


/** convention is c = [theta, phi], \theta \in [0, \pi], \phi \in [0, 2\pi] and y is up as in theta = 0 => direction = [0, 1, 0]  */
class SphericalCoordinates
{
public:

  struct ToDirectionResult
  {
    float cosTheta;
    float sinTheta;
    Vec3f direction;
  };
  static ToDirectionResult ToDirection(const Vec2f& c)
  {
    ToDirectionResult result;
    result.cosTheta = std::cos(c[0]);
    result.sinTheta = std::sin(c[0]);
    float sinphi = std::sin(c[1]); 
    float cosphi = std::cos(c[1]);
    result.direction = Vec3f(result.sinTheta * cosphi, result.cosTheta, result.sinTheta * sinphi);
    return result;
  }
  static Vec2f FromDirection(const Vec3f& direction)
  {
    assertUnitLength(direction);
    float theta = std::acos(direction[1]);
    float phi = std::atan2(direction[2], direction[0]);
    phi = phi < 0.0f ? phi + float(2.0 * Hasty::Pi) : phi;
    return Vec2f(theta, phi);
  }
};

inline float pow(float v, float exponent)
{
  return std::powf(v, exponent);
}
inline Vec3f pow(const Vec3f& v, float exponent)
{
  return v.cwisePow(exponent);
}

inline float clamp(float v, float low, float high)
{
  return v < low ? low : (v > high ? high : v);
}
inline Vec3f clamp(const Vec3f& v, float low, float high)
{
  return v.clamp(low, high);
}

}
