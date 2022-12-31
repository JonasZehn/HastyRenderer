#pragma once

#include <nlohmann/json.hpp>

#include <cmath>
#include <array>
#include <iostream>
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
  static_assert(N >= 0, "only support non negative exponents, use pow instead");
  if constexpr(N == 0) return 1.0f;
  else
  {
    float q = powci<N / 2>(f);
    if constexpr(N % 2 == 1)
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
  inline Vec2f()
  {
#ifdef INITIALIZE_WITH_NAN
    m_data[0] = std::numeric_limits<float>::signaling_NaN();
    m_data[1] = std::numeric_limits<float>::signaling_NaN();
#endif
  }
  inline Vec2f(float x, float y)
  {
    m_data[0] = x;
    m_data[1] = y;
  }
  inline float& operator[](std::size_t i)
  {
    return m_data[i];
  }
  inline const float& operator[](std::size_t i) const
  {
    return m_data[i];
  }
  inline Vec2f operator+(const Vec2f& v2) const
  {
    return Vec2f(m_data[0] + v2[0], m_data[1] + v2[1]);
  }
  inline Vec2f operator-(const Vec2f& v2) const
  {
    return Vec2f(m_data[0] - v2[0], m_data[1] - v2[1]);
  }
  inline Vec2f operator*(float f) const
  {
    return Vec2f(m_data[0] * f, m_data[1] * f);
  }
  inline Vec2f& operator+=(const Vec2f& v2)
  {
    m_data[0] += v2[0];
    m_data[1] += v2[1];
    return *this;
  }
  inline float dot(const Vec2f& v2) const
  {
    return m_data[0] * v2[0] + m_data[1] * v2[1];
  }
  inline float normSq() const
  {
    return this->dot(*this);
  }
  inline float norm() const
  {
    return std::sqrt(this->dot(*this));
  }
  inline static Vec2f Constant(float f)
  {
    return Vec2f(f, f);
  }
  inline static Vec2f Zero()
  {
    return Constant(0.0f);
  }
  inline static Vec2f Ones()
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

  inline Vec3f()
  {
#ifdef INITIALIZE_WITH_NAN
    m_data[0] = std::numeric_limits<float>::signaling_NaN();
    m_data[1] = std::numeric_limits<float>::signaling_NaN();
    m_data[2] = std::numeric_limits<float>::signaling_NaN();
#endif
  }
  inline Vec3f(float x, float y, float z)
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
  }
  inline float& operator[](std::size_t i)
  {
    return m_data[i];
  }
  inline const float& operator[](std::size_t i) const
  {
    return m_data[i];
  }
  inline Vec3f operator+(const Vec3f& v2) const
  {
    return Vec3f(m_data[0] + v2[0], m_data[1] + v2[1], m_data[2] + v2[2]);
  }
  inline Vec3f& operator+=(const Vec3f& v2)
  {
    m_data[0] += v2[0];
    m_data[1] += v2[1];
    m_data[2] += v2[2];
    return *this;
  }
  inline Vec3f operator/(float s) const
  {
    return Vec3f(m_data[0] / s, m_data[1] / s, m_data[2] / s);
  }
  inline Vec3f& operator*=(float s)
  {
    m_data[0] *= s;
    m_data[1] *= s;
    m_data[2] *= s;
    return *this;
  }
  inline Vec3f& operator/=(float s)
  {
    m_data[0] /= s;
    m_data[1] /= s;
    m_data[2] /= s;
    return *this;
  }
  inline Vec3f operator-() const
  {
    return Vec3f(-m_data[0], -m_data[1], -m_data[2]);
  }
  inline Vec3f operator-(const Vec3f& v2) const
  {
    return Vec3f(m_data[0] - v2[0], m_data[1] - v2[1], m_data[2] - v2[2]);
  }
  inline Vec3f operator*(float s) const
  {
    return Vec3f(m_data[0] * s, m_data[1] * s, m_data[2] * s);
  }
  inline bool operator==(const Vec3f& v2) const
  {
    return m_data[0] == v2[0] && m_data[1] == v2[1] && m_data[2] == v2[2];
  }
  inline bool operator!=(const Vec3f& v2) const
  {
    return !this->operator==(v2);
  }

  inline static Vec3f Fill(float f)
  {
    return Vec3f(f, f, f);
  }
  inline static Vec3f Zero()
  {
    return Fill(0.0f);
  }
  inline static Vec3f Ones()
  {
    return Fill(1.0f);
  }

  inline std::size_t size() const
  {
    return m_data.size();
  }

private:
  std::array<float, 3> m_data;
};

inline Vec3f cross(const Vec3f& v1, const Vec3f& v2)
{
  return Vec3f(
    v1[1] * v2[2] - v1[2] * v2[1],
    v1[2] * v2[0] - v1[0] * v2[2],
    v1[0] * v2[1] - v1[1] * v2[0]
  );
}
inline Vec3f abs(const Vec3f& v1)
{
  return Vec3f(std::abs(v1[0]), std::abs(v1[1]), std::abs(v1[2]));
}
inline Vec3f cwiseProd(const Vec3f& v1, const Vec3f& v2)
{
  return Vec3f(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}
inline Vec3f exp(const Vec3f& v1)
{
  return Vec3f(std::exp(v1[0]), std::exp(v1[1]), std::exp(v1[2]));
}
inline Vec3f log(const Vec3f& v1)
{
  return Vec3f(std::log(v1[0]), std::log(v1[1]), std::log(v1[2]));
}
inline Vec3f sqrt(const Vec3f& v1)
{
  return Vec3f(std::sqrt(v1[0]), std::sqrt(v1[1]), std::sqrt(v1[2]));
}
inline Vec3f pow(const Vec3f& v1, float exponent)
{
  return Vec3f(std::pow(v1[0], exponent), std::pow(v1[1], exponent), std::pow(v1[2], exponent));
}
inline bool isFinite(const Vec3f& v1)
{
  return std::isfinite(v1[0]) && std::isfinite(v1[1]) && std::isfinite(v1[2]);
}

inline Vec3f clamp(const Vec3f& v1, float bottom, float top)
{
  return Vec3f(
    std::max(bottom, std::min(top, v1[0])),
    std::max(bottom, std::min(top, v1[1])),
    std::max(bottom, std::min(top, v1[2]))
  );
}
inline float dot(const Vec3f& v1, const Vec3f& v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}
inline float normSq(const Vec3f& v1)
{
  return dot(v1, v1);
}
inline float norm(const Vec3f& v1)
{
  return std::sqrt(normSq(v1));
}
inline Vec3f normalize(const Vec3f& v1)
{
  return v1 / norm(v1);
}
inline float sum(const Vec3f& v1)
{
  return v1[0] + v1[1] + v1[2];
}
inline float normL1(const Vec3f& v1)
{
  return sum(abs(v1));
}
inline float mix(float v1, float v2, float lambda)
{
  return (1.0f - lambda) * v1 + lambda * v2;
}
inline Vec3f mix(const Vec3f& v1, const Vec3f& v2, float lambda)
{
  Vec3f result;
  for(int i = 0; i < result.size(); i++)
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
  inline Vec4f()
  {
#ifdef INITIALIZE_WITH_NAN
    m_data[0] = std::numeric_limits<float>::signaling_NaN();
    m_data[1] = std::numeric_limits<float>::signaling_NaN();
    m_data[2] = std::numeric_limits<float>::signaling_NaN();
    m_data[3] = std::numeric_limits<float>::signaling_NaN();
#endif
  }
  inline Vec4f(float x, float y, float z, float w)
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = w;
  }
  inline Vec4f& operator+=(const Vec4f& v2)
  {
    m_data[0] += v2[0];
    m_data[1] += v2[1];
    m_data[2] += v2[2];
    m_data[3] += v2[3];
    return *this;
  }
  inline Vec4f operator/(float s) const
  {
    return Vec4f(m_data[0] / s, m_data[1] / s, m_data[2] / s, m_data[3] / s);
  }
  inline Vec4f& operator/=(float s)
  {
    m_data[0] /= s;
    m_data[1] /= s;
    m_data[2] /= s;
    m_data[3] /= s;
    return *this;
  }
  inline float& operator[](std::size_t i)
  {
    return m_data[i];
  }
  inline const float& operator[](std::size_t i) const
  {
    return m_data[i];
  }

  inline static Vec4f Constant(float f)
  {
    return Vec4f(f, f, f, f);
  }
  inline static Vec4f Zero()
  {
    return Constant(0.0f);
  }
  inline static Vec4f Ones()
  {
    return Constant(1.0f);
  }


private:
  std::array<float, 4> m_data;
};

inline Vec4f clamp(const Vec4f& v1, float bottom, float top)
{
  return Vec4f(
    std::max(bottom, std::min(top, v1[0])),
    std::max(bottom, std::min(top, v1[1])),
    std::max(bottom, std::min(top, v1[2])),
    std::max(bottom, std::min(top, v1[3]))
  );
}

inline void assertUnitLength(const Vec3f& v)
{
  assert(std::abs(norm(v) - 1) < 1e-4f);
}

inline void assertFinite(const Vec3f& v)
{
  assert(isFinite(v));
}
inline void assertFinite(float v)
{
  assert(std::isfinite(v));
}

// col major
class Mat2f
{
public:
  inline Mat2f()
  {
#ifdef INITIALIZE_WITH_NAN
    m_data[0] = std::numeric_limits<float>::signaling_NaN();
    m_data[1] = std::numeric_limits<float>::signaling_NaN();
    m_data[2] = std::numeric_limits<float>::signaling_NaN();
    m_data[3] = std::numeric_limits<float>::signaling_NaN();
#endif
  }
  inline Mat2f(const Vec2f& col1, const Vec2f& col2)
  {
    this->operator()(0, 0) = col1[0];
    this->operator()(1, 0) = col1[1];
    this->operator()(0, 1) = col2[0];
    this->operator()(1, 1) = col2[1];
  }

  inline Mat2f inverse() const
  {
    Mat2f result;
    float d = this->operator()(0, 0) * this->operator()(1, 1) - this->operator()(0, 1) * this->operator()(1, 0);
    result(0, 0) = this->operator()(1, 1) / d;
    result(0, 1) = -this->operator()(0, 1) / d;
    result(1, 0) = -this->operator()(1, 0) / d;
    result(1, 1) = this->operator()(0, 0) / d;
    return result;
  }

  inline float& operator()(int i, int j)
  {
    return m_data[i + 2 * j];
  }
  inline float operator()(int i, int j) const
  {
    return m_data[i + 2 * j];
  }

private:
  std::array<float, 4> m_data;
};

inline Vec3f orthonormalized(const Vec3f& v1Normalized, const Vec3f& v2)
{
  assertUnitLength(v1Normalized);
  Vec3f result = v2 - dot(v2, v1Normalized) * v1Normalized;
  float length = norm(result);
  assert(length != 0.0f);
  result /= length;
  return result;
}

inline Vec3f anyOrthonormal(const Vec3f& v1Normalized)
{
  assertUnitLength(v1Normalized);
  if(std::abs(v1Normalized[0]) < 0.7f)
  {
    return orthonormalized(v1Normalized, Vec3f(1.0f, 0.0f, 0.0f));
  }
  else
  {
    return orthonormalized(v1Normalized, Vec3f(0.0f, 1.0f, 0.0f));
  }
}

inline Vec3f orthonormalizedOtherwiseAnyOrthonormal(const Vec3f& v1Normalized, const Vec3f& v2)
{
  assertUnitLength(v1Normalized);
  Vec3f result = v2 - dot(v2, v1Normalized) * v1Normalized;
  float length = norm(result);
  if(length < 1e-7f)
  {
    result = anyOrthonormal(v1Normalized);
  }
  else
  {
    result /= length;
  }
  return result;
}

struct templateDummyType {};

template<typename T>
inline T zero()
{
  static_assert(!std::is_same<T, templateDummyType>::value, "this function is not supposed to work, see other specializations");
}

template <>
inline int32_t zero()
{
  return static_cast<int32_t>(0);
}
template <>
inline float zero()
{
  return 0.0f;
}
template <>
inline Vec2f zero()
{
  return Vec2f::Zero();
}
template <>
inline Vec3f zero()
{
  return Vec3f::Zero();
}
template <>
inline Vec4f zero()
{
  return Vec4f::Zero();
}


template<typename T>
inline T one()
{
  static_assert(!std::is_same<T, templateDummyType>::value, "this function is not supposed to work, see other specializations");
}

template <>
inline int32_t one()
{
  return static_cast<int32_t>(1);
}
template <>
inline float one()
{
  return 1.0f;
}
template <>
inline Vec2f one()
{
  return Vec2f::Ones();
}
template <>
inline Vec3f one()
{
  return Vec3f::Ones();
}
template <>
inline Vec4f one()
{
  return Vec4f::Ones();
}


template<class ScalarType, class VectorType>
class RotationBetweenTwoVectors
{
public:
  template<class S, class V>
  friend V applyRotation(const RotationBetweenTwoVectors<S, V>& rotation, const V& x);
  template<class S, class V>
  friend V applyRotationInverse(const RotationBetweenTwoVectors<S, V>& rotation, const V& x);

  RotationBetweenTwoVectors(const VectorType& vFrom, const VectorType& vTo)
  {
    assertUnitLength(vFrom);
    assertUnitLength(vTo);
    ScalarType cosAngle = dot(vFrom, vTo);
    ScalarType onePlusCosAngle = ScalarType(1.0) + cosAngle;
    if(std::abs(onePlusCosAngle) > std::numeric_limits<ScalarType>::epsilon())
    {
      // v_rot = x * cos(angle) + (k sin(angle) cross x + (k sin(angle) )  ( (k sin(angle) ) dot x ) / (1 + cos(angle))
      VectorType kSinAngle = cross(vFrom, vTo);
      a = cosAngle;
      b = kSinAngle;
      c = kSinAngle;
      d = kSinAngle / onePlusCosAngle;
    }
    else
    {
      VectorType k = anyOrthonormal(vFrom); // find rotation axis so we don't invert volume
      a = ScalarType(-1.0);
      b = zero<VectorType>();
      c = k;
      d = k * ScalarType(2.0); // k* (ScalarType(1.0) - cosAngle);
    }
  }

private:
  ScalarType a;
  VectorType b;
  VectorType c;
  VectorType d;
};
template<class ScalarType, class VectorType>
inline VectorType applyRotation(const RotationBetweenTwoVectors<ScalarType, VectorType>& rotation, const VectorType& x)
{
  return x * rotation.a + cross(rotation.b, x) + rotation.c * dot(rotation.d, x);
}
template<class ScalarType, class VectorType>
inline VectorType applyRotationInverse(const RotationBetweenTwoVectors<ScalarType, VectorType>& rotation, const VectorType& x)
{
  return x * rotation.a - cross(rotation.b, x) + rotation.c * dot(rotation.d, x);
}

class Ray
{
public:
  inline Ray() {}
  inline Ray(const Vec3f& origin, const Vec3f& direction)
  {
    m_origin = origin;
    m_direction = direction;
  }
  inline const Vec3f& origin() const
  {
    return m_origin;
  }
  inline const Vec3f& direction() const
  {
    return m_direction;
  }

private:
  Vec3f m_origin;
  Vec3f m_direction;

};


/**
* incident N
*    |\` /|\
*       \ |
* ------------------
*       /
*     |/_
*   return value
* or
*         N    / return value
*        /|\  /
*         | /
* ------------------
*          \
*           \
*            _| incident
*
*/
inline Vec3f reflect(const Vec3f& incident, const Vec3f& N)
{
  return incident - 2.0f * dot(N, incident) * N;
}
/**
* outg    N   return value
*    |\` /|\ `/|
*       \ | /
* ------------------
*          \
*           _\|
*            - outg
*/
inline Vec3f reflectAcross(const Vec3f& outgoing, const Vec3f& N)
{
  return reflect(-outgoing, N); // -outgoing + 2.0f * N.dot(outgoing) * N;
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

inline float log(float v)
{
  return std::log(v);
}
inline float sqrt(float v)
{
  return std::sqrt(v);
}

inline float pow(float v, float exponent)
{
  return std::pow(v, exponent);
}

inline float clamp(float v, float low, float high)
{
  return v < low ? low : (v > high ? high : v);
}
inline bool isFinite(float v1)
{
  return std::isfinite(v1);
}

}
