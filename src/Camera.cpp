#include <Hasty/Camera.h>

#include <Hasty/Sample.h>

namespace Hasty
{


void from_json(const nlohmann::json& j, Camera& camera)
{
  Vec3f pos = j.at("position").get<Vec3f>();
  if(j.find("look_at") != j.end())
  {
    camera.setLookAt(pos, j.at("look_at").get<Vec3f>());
  }
  else
  {
    camera.setLookAt(pos, pos + Vec3f(0.0f, 0.0f, -1.0f));
  }
  json_get_optional(camera.m_apertureSize, j, "aperture", 0.0f);
  json_get_optional(camera.m_focalDistance, j, "focal_distance", 1.0f);
  json_get_optional(camera.m_exposure, j, "exposure", 0.0f);
  json_get_optional(camera.m_numBlades, j, "num_blades", int(8));
  json_get_optional(camera.m_bladeRotation, j, "blade_rotation", 0.0f);
}

Camera::Camera()
{
}
void Camera::setLookAt(const Vec3f& position, const Vec3f& lookAt)
{
  m_position = position;
  m_forward = normalize(lookAt - position);
  m_up = Vec3f(0.0f, 1.0f, 0.0f);
  //orthogonalize
  m_up = m_up - dot(m_up, m_forward) * m_forward;
  m_up = normalize(m_up);
  m_right = cross(m_forward, m_up);
}
// x goes to right, y goes to the bottom
Ray Camera::computeRay(RNG& rng, const Vec2f& p, float frameWidth, float frameHeight)
{
  bool depthOfField = m_apertureSize > 0.0f;
  if(depthOfField)
  {
    // we aren't doing  a "physical" lense for now
     // focal plane is at m_position + m_forward * m_focal distance
    // we move origin of the ray randomly, but on the focal plane these rays have to hit the same point independent of the random offset
    float x = p[0];
    float y = p[1];
    float fovDegree = m_fovDegree;
    float fovRadians = float(Pi) * fovDegree / 180.0f;
    float q = std::tan(fovRadians * 0.5f); // fov is defined as angle across both directions thus we have to divide by 2 here
    float lx = (x - 0.5f * frameWidth) / frameWidth * 2.0f;
    float ly = (y - 0.5f * frameHeight) / frameWidth * 2.0f;
    Vec3f direction = m_forward + m_right * (q * lx) - m_up * (q * ly);
    direction = normalize(direction);
    float density;
    Vec2f offset = sampleCircularNGonUniformly(rng, m_numBlades, m_bladeRotation, density);
    float t = m_focalDistance * dot(direction, m_forward);
    Vec3f focalPoint = m_position + t * direction;
    Vec3f origin = m_position + m_apertureSize * offset[0] * m_up + m_apertureSize * offset[1] * m_right;
    // now adapt the direction
    direction = normalize(focalPoint - origin);

    return Ray(origin, direction);
  }
  else
  {
    float x = p[0];
    float y = p[1];
    float fovDegree = m_fovDegree;
    float fovRadians = float(Pi) * fovDegree / 180.0f;
    float q = std::tan(fovRadians * 0.5f); // fov is defined as angle across both directions thus we have to divide by 2 here
    float lx = (x - 0.5f * frameWidth) / frameWidth * 2.0f;
    float ly = (y - 0.5f * frameHeight) / frameWidth * 2.0f;
    Vec3f origin = m_position;
    Vec3f direction = m_forward + m_right * (q * lx) - m_up * (q * ly);
    direction = normalize(direction);
    return Ray(origin, direction);
  }
}
float Camera::computeRayAngle(const Vec2f& p, float frameWidth, float frameHeight)
{
  float x = p[0];
  float y = p[1];
  float fovDegree = m_fovDegree;
  float fovRadians = float(Pi) * fovDegree / 180.0f;
  float q = std::tan(fovRadians * 0.5f); // fov is defined as angle across left to right (not to middle) thus we have to divide by 2 here
  float lx1 = (x + 0.5f - 0.5f * frameWidth) / frameWidth * 2.0f;
  float ly1 = (y + 0.5f - 0.5f * frameHeight) / frameWidth * 2.0f;
  float lx2 = (x - 0.5f - 0.5f * frameWidth) / frameWidth * 2.0f;
  float ly2 = (y - 0.5f - 0.5f * frameHeight) / frameWidth * 2.0f;

  float angleX = std::abs(std::atan(q * lx1) - std::atan(q * lx2));
  float angleY = std::abs(std::atan(q * ly1) - std::atan(q * ly2));
  return 0.5f * angleX + 0.5f * angleY;
}

}