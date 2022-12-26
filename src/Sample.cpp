#include <Hasty/Sample.h>

namespace Hasty
{
Vec2f sampleDiskUniformly(RNG& rng, float* pDensity)
{
  float alpha = uniform01f(rng) * float(2.0 * Hasty::Pi);
  float r = std::sqrt(uniform01f(rng));
  (*pDensity) = float(Hasty::InvPi);
  return Vec2f(r * std::cos(alpha), r * std::sin(alpha));
}
Vec2f sampleCircularNGonUniformly(RNG& rng, int numBlades, float bladeRotation, float* pDensity)
{
  assert(bladeRotation >= float(-Hasty::Pi));
  assert(numBlades >= 3);
  
  // do a half plane test with direction of anglePerSector/2
  float anglePerSector = float(2.0 * Hasty::Pi / numBlades);
  Vec2f normal(std::cos(anglePerSector * 0.5f), std::sin(anglePerSector * 0.5f));
  float d = -normal[0];  // offset such that (1,0) is on the border of the half plane

  Vec2f sample;
  bool outside;
  do
  {
    sample = sampleDiskUniformly(rng, pDensity);
    float r = sample.norm();
    float angle = std::atan2(sample[1], sample[0]);
    angle -= bladeRotation;
    angle += float(2.0 * Hasty::Pi)* (1.0f + std::numeric_limits<float>::epsilon()); // make sure angle is bigger than zero
    // now normalize wrt to angular sector:
    angle = std::fmod(angle, anglePerSector);
    Vec2f normalizedPosition(r * std::cos(angle), r * std::sin(angle));
    outside = (normal.dot(normalizedPosition) + d) > 0.0f;
  } while (outside);

  float areaSector = 0.5 * std::sin(anglePerSector); // Area = 0.5 * radius * radius * sin(anglePerSector)
  float areaNGon = numBlades * areaSector;
  (*pDensity) = 1.0 / areaNGon;

  return sample;
}
Vec3f sampleSphereSurfaceUniformly(RNG& rng, float* pDensity)
{
  (*pDensity) = 0.25f * float(InvPi); // 4 pi = Area of sphere with radius 1
  while (true)
  {
    Vec3f r(rng.uniformm11f(), rng.uniformm11f(), rng.uniformm11f());
    float rsq = r.dot(r);
    if (rsq <= 1.0 && rsq > 1e-6f)
    {
      return r / std::sqrt(rsq);
    }
  }
  return Vec3f::Zero();
}


Vec3f sampleHemisphereSurfaceUniformly(RNG& rng, const Vec3f& normal, float* pDensity)
{
  (*pDensity) = 0.5f * float(InvPi); // 2 pi = Area of hemisphere with radius 1
  while (true)
  {
    Vec3f r(rng.uniformm11f(), rng.uniformm11f(), rng.uniformm11f());
    float rsq = r.dot(r);
    if (rsq <= 1.0f && rsq > 1e-6f)
    {
      if (r.dot(normal) >= 0.0f)
      {
        return r / std::sqrt(rsq);
      }
    }
  }
  return Vec3f::Zero();
}

Vec3f sampleHemisphereCosImportance(RNG& rng, const Vec3f& normal, float* pDensity)
{
  // we want to imporance sample \omega such that \omega ~ n.dot(\omega) = cos(theta)
  // since d\omega = sin(\theta) d\theta d\phi , we need (\phi, \theta) ~ (uniform, sin(\theta) cos(\theta) )
  // Inverse transform sampling : 
  // P(\Theta < \theta) = \int_0^\theta 1/C sin(\theta') cos(\theta') d\theta' = sin^2(\theta')/(2C)        P(\Theta <= \pi/2) = 1/(2C) = 1 => C = 1/2
  //                     = sin^2(\theta') = 1 - cos^2(\theta')
  // \theta = F^{-1}(u),  F(\theta) = u = 1 - cos^2(\theta') <=> \theta' = acos(sqrt(1 - u)), and now  this is equivalent to acos(sqrt(u)) because u is uniformly sampled
  Vec3f dir;
  float costheta;
  do
  {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float xi1 = uniform01f(rng);
    float xi2 = uniform01f(rng);

    float phi = 2.0f * float(Pi) * xi2;
    costheta = std::sqrt(xi1);
    float sintheta = std::sqrt(1.0f - xi1); // std::sqrt(1.0f - costheta * costheta);
    float cosphi = std::cos(phi);
    float sinphi = std::sin(phi); // you cannot use sqrt(1 - cosphi*cosphi) here you are gonna loose the sign
    Vec3f rv(sintheta * cosphi, sintheta * sinphi, costheta);

    RotationBetweenTwoVectors<float, Vec3f> rotation(Vec3f(0.0f, 0.0f, 1.0f), normal);
    dir = rotation * rv;
    dir.normalize();
    costheta = dir.dot(normal);
  } while (costheta <= 0.0f); // stay clear from numerical mistakes

  (*pDensity) = float(InvPi) * costheta;
  return dir;
}
float evaluateHemisphereCosImportancePDF(const Vec3f& normal, const Vec3f& direction)
{
  float cosTheta = normal.dot(direction);
  if (cosTheta < 0.0f) return 0.0f;
  return float(InvPi) * cosTheta;
}

// https://cseweb.ucsd.edu/~viscomp/classes/cse168/sp21/lectures/168-lecture9.pdf
// https://alexanderameye.github.io/notes/sampling-the-hemisphere/
// https://digibug.ugr.es/bitstream/handle/10481/19751/rmontes_LSI-2012-001TR.pdf
Vec3f sampleCosineLobe(RNG& rng, const Vec3f& lobeDirection, float exponent, float* pDensity)
{
  // here we just sample according to  \theta ~ cos(\theta)^exponent sin(\theta), and then the vector is transformed into the lobe frame
  // this doesn't create directions on the whole hemisphere wrt the normal!
  //  F(\theta) = \int_0^\theta 1/C cos(\theta')^n sin(\theta') d\theta = [ - 1/C cos^{n+1}(\theta')/(n+1) ]_0^\theta = [ - 1/C cos^{n+1}(\theta)/(n+1) + 1/C /(n+1) ]
  //   F(\pi / 2) = 1 = + 1/C /(n+1) => C = 1/(n+1)
  // F(\theta) = u = 1 - cos^{n+1}(\theta) => \theta = acos( (1 - u)^{1/(n+1)} ) which is "equivalent" to acos( u^{1/(n+1)} )
  Vec3f dir;
  float costheta;
  do
  {
    float xi1 = uniform01f(rng);
    float xi2 = uniform01f(rng);

    float phi = 2.0f * float(Pi) * xi2;
    costheta = std::pow(xi1, 1.0f / (exponent + 1.0f));
    float sintheta = std::sqrt(1.0f - costheta * costheta);
    float cosphi = std::cos(phi);
    float sinphi = std::sin(phi); // you cannot use sqrt(1 - cosphi*cosphi) here you are gonna loose the sign
    Vec3f rv(sintheta * cosphi, sintheta * sinphi, costheta);

    RotationBetweenTwoVectors<float, Vec3f> rotation(Vec3f(0.0f, 0.0f, 1.0f), lobeDirection);
    dir = rotation * rv;
    dir.normalize();
    costheta = dir.dot(lobeDirection);
  } while (costheta <= 0.0f); // stay clear from numerical mistakes

  (*pDensity) = (exponent + 1.0f) * 0.5f * float(InvPi) * std::pow(costheta, exponent);
  return dir;
}

float evaluateCosineLobePDF(const Vec3f& lobeDirection, float exponent, const Vec3f& direction)
{
  float cosTheta = lobeDirection.dot(direction);
  if (cosTheta < 0.0f) return 0.0f;
  return (exponent + 1.0f) * 0.5f * float(InvPi) * std::pow(cosTheta, exponent);
}

Vec3f sampleTriangleUniformly(RNG& rng)
{
  // https://math.stackexchange.com/questions/1785136/generating-randomly-distributed-points-inside-a-given-triangle
  // [s,q] ~ U([0, 1]^2)
  // t = sqrt(q)
  // Q_s = (1 - s) P_2 + s P_3
  // P_{s,q} = (1 - t) P_1 + t Q_s
  //         = (1 - sqrt(q) ) P_1 + sqrt(q) (1 - s) P_2 + sqrt(q) s P_3
  // f(x, y) = f(s, q) / abs(detjac(xy))
  // we choose P_2 = 0 , without loss of generality
  // d xy / ds = sqrt(q) P_3
  // d xy / dq = 1/(2 sqrt(q) ) ( s P_3 - P_1)
  // => detjac = p3x / 2 ( s p3y - p1y) - p3y / 2 ( s p3x - p1x) = - p1y p3x / 2   + p1x p3y / 2 
  float s = uniform01f(rng);
  float t = std::sqrt(uniform01f(rng));
  float beta1 = 1.0f - t;
  float beta3 = t * s;
  float beta2 = std::max(0.0f, 1.0f - beta1 - beta3);
  Vec3f barycentricCoordinates(beta1, beta2, beta3);
  return barycentricCoordinates;
}

}

