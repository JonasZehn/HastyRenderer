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
  Ray ray(RNG &rng, const Vec2f& p, float frameWidth, float frameHeight);
  float rayAngle(const Vec2f& p, float frameWidth, float frameHeight);
  const Vec3f& position() const { return m_position; }

  float exposure() const { return m_exposure; }
private:
  Vec3f m_position = Vec3f(0.f, 0.f, 0.f);
  Vec3f m_forward = Vec3f(0.f, 0.f, -1.f);
  Vec3f m_right = Vec3f(1.f, 0.f, 0.f);
  Vec3f m_up = m_right.cross(m_forward);
  float m_fovDegree = 45.0f;
  float m_apertureSize = 0.0f;
  float m_focalDistance = 1.0f;
  float m_exposure = 0.0f;
};


}
