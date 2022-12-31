#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Random.h>
#include <Hasty/Json.h>

namespace Hasty
{

class Camera;

void from_json(const nlohmann::json& j, Camera& p);
class Camera
{
public:
  friend void from_json(const nlohmann::json& j, Camera& p);

  Camera();

  void setLookAt(const Vec3f& position, const Vec3f& lookAt);
  // x goes to right, y goes to the bottom
  Ray computeRay(RNG& rng, const Vec2f& p, float frameWidth, float frameHeight);
  float computeRayAngle(const Vec2f& p, float frameWidth, float frameHeight);
  const Vec3f& getPosition() const { return m_position; }
  float getFoVSlope() { return std::tan(0.5f * (float(Pi) * m_fovDegree / 180.0f)); }
  Vec3f getForward() const { return m_forward; }
  Vec3f getUp() const { return m_up; }
  Vec3f getRight() const { return m_right; }

  float getExposure() const { return m_exposure; }

private:
  Vec3f m_position = Vec3f(0.f, 0.f, 0.f);
  Vec3f m_forward = Vec3f(0.f, 0.f, -1.f);
  Vec3f m_right = Vec3f(1.f, 0.f, 0.f);
  Vec3f m_up = cross(m_right, m_forward);
  float m_fovDegree = 45.0f;
  float m_apertureSize = 0.0f;
  float m_focalDistance = 1.0f;
  float m_exposure = 0.0f;
  int m_numBlades = 8;
  float m_bladeRotation = 0.0f;
};


}
