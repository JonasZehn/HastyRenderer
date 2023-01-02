#include <gtest/gtest.h>

#include <Hasty/BRDF.h>
#include <Hasty/VMath.h>
#include <Hasty/Random.h>
#include <Hasty/Scene.h>

#include <mutex>

using namespace Hasty;

#define EXPECT_NEARREL(a, b, c)  EXPECT_NEAR(a, b, std::max(1.0f, std::abs(a) ) * c );  EXPECT_NEAR(a, b, std::max(1.0f,   std::abs(b)  ) * c );

std::mutex g_mapLock;
class SingleThreadedTestHelper
{
public:
  SingleThreadedTestHelper(std::string name)
    :m_lock(getLock(name))
  {
  }
  ~SingleThreadedTestHelper()
  {
    m_lock.unlock();
  }

  static std::mutex& getLock(const std::string& n)
  {
    g_mapLock.lock();
    static std::map<std::string, std::mutex> mutexMap;
    std::mutex &result = mutexMap[n];
    result.lock();
    g_mapLock.unlock();
    return result;
  }
private:
  std::mutex &m_lock;
};

TEST(vmath, rodriguesRotation)
{
  RNG rng(0);

  float pDensity;
  RotationBetweenTwoVectors<float, Vec3f> rot(sampleSphereSurfaceUniformly(rng, pDensity), sampleSphereSurfaceUniformly(rng, pDensity));
  Vec3f x0 = rng.uniform01Vec3f();
  Vec3f x1 = applyRotation(rot, x0);
  Vec3f x0prime = applyRotationInverse(rot, x1);
  EXPECT_NEAR(x0[0], x0prime[0], 1e-4f);
  EXPECT_NEAR(x0[1], x0prime[1], 1e-4f);
  EXPECT_NEAR(x0[2], x0prime[2], 1e-4f);
}

TEST(vmath, sphericalCoordinates)
{
  RNG rng(0);
  
  for (int i = 0; i < 50; i++)
  {
    float pDensity;
    Vec3f dir = sampleSphereSurfaceUniformly(rng, pDensity);
    auto fromResult = SphericalCoordinates::FromDirection(dir);
    auto toResult = SphericalCoordinates::ToDirection(fromResult);
    
    EXPECT_TRUE(fromResult[0] >= 0.0f && fromResult[0] <= float(Hasty::Pi));
    EXPECT_TRUE(fromResult[1] >= 0.0f && fromResult[1] <= float(2.0 * Hasty::Pi));

    EXPECT_NEAR(dir[0], toResult.direction[0], 1e-4f);
    EXPECT_NEAR(dir[1], toResult.direction[1], 1e-4f);
    EXPECT_NEAR(dir[2], toResult.direction[2], 1e-4f);
  }


}

TEST(brdf, sampleCosineLobe)
{
  RNG rng(0);

  for (int i = 0; i < 50; i++)
  {
    float pDensity;
    Vec3f lobeDirection = sampleSphereSurfaceUniformly(rng, pDensity);
    float exponent = 1.0f + uniform01f(rng) * 10.0f;
    Vec3f wi = sampleCosineLobe(rng, lobeDirection, exponent, pDensity);
    EXPECT_NEARREL(norm(wi), 1.0f, 1e-5f);
    float pDensity2 = evaluateCosineLobePDF(lobeDirection, exponent, wi);
    EXPECT_NEARREL(pDensity, pDensity2, 1e-5f );
  }
}


TEST(brdf, sampleHemisphereCosImportance)
{
  RNG rng(0);

  for (int i = 0; i < 50; i++)
  {
    float pDensity;
    Vec3f normal = sampleSphereSurfaceUniformly(rng, pDensity);
    Vec3f wi = sampleHemisphereCosImportance(rng, normal, pDensity);
    EXPECT_NEARREL(norm(wi), 1.0f, 1e-4f);
    float pDensity2 = evaluateHemisphereCosImportancePDF(normal, wi);
    EXPECT_NEARREL(pDensity, pDensity2, 1e-4f );
  }
}
TEST(brdf, DGGX)
{
  RNG rng(0);

  //see Microfacet Models for Refraction through Rough Surfaces for a description of some properties
  float sum = 0.0f;
  float n = 50000;
  float alpha = 0.3f;
  float pDensity;
  Vec3f normal = sampleSphereSurfaceUniformly(rng, pDensity);
  Vec3f wo = sampleHemisphereSurfaceUniformly(rng, normal, pDensity);
  Vec3f tangent = anyOrthonormal(normal);
  Vec3f bitangent = cross(normal, tangent);
  for (int i = 0; i < n; i++)
  {
    //Vec3f wi = sampleGGXVNDFGlobal(rng, normal, alpha, wo, pDensity);
    wo = sampleHemisphereSurfaceUniformly(rng, normal, pDensity);
    float d = DGGX(dot(normal, wo), dot(tangent, wo), dot(bitangent, wo), alpha, alpha);
    EXPECT_TRUE(d >= 0.0f);
    sum += d * dot(normal, wo) / pDensity;
  }
  float estimate = sum / n;
  
  EXPECT_NEARREL(estimate, 1.0f, 1e-2f );
}
TEST(brdf, sampleDGGX)
{
  RNG rng(0);

  for (int i = 0; i < 50; i++)
  {
    float alpha = std::max(1e-2f, uniform01f(rng));
    float pDensity;
    Vec3f normal = sampleSphereSurfaceUniformly(rng, pDensity);
    Vec3f wo = sampleHemisphereSurfaceUniformly(rng, normal, pDensity);
    Vec3f wi = sampleDGGX(rng, normal, alpha, wo, pDensity);
    EXPECT_NEARREL(norm(wi), 1.0f, 1e-4f);
    float pDensity2 = sampleDGGXDensity(normal, alpha, wo, wi);
    EXPECT_NEARREL(pDensity, pDensity2, 1e-2f );
  }
}
TEST(brdf, sampleDGGX2)
{
  RNG rng(0);
  
  float alpha = 1.0f;
  float pDensity;
  Vec3f normal = sampleSphereSurfaceUniformly(rng, pDensity);
  // we can only use this as a sampling distribution if we choose wo = normal, otherwise not the whole hemisphere is properly samples....
  Vec3f wo = normal;

  float sum = 0.0f;
  float n = 500000;
  for (int i = 0; i < n; i++)
  {
    Vec3f wi = sampleDGGX(rng, normal, alpha, wo, pDensity);
    EXPECT_TRUE(pDensity >= 0.0f);
    sum += std::max(0.0f, dot(normal, wi)) / pDensity;
  }
  float estimate = sum / n;
  EXPECT_NEARREL(estimate, float(Pi), 1e-2f );
}
TEST(brdf, sampleGGXVNDFGlobal)
{
  RNG rng(0);

  for (int i = 0; i < 50; i++)
  {
    float alpha = std::max(1e-2f, uniform01f(rng));
    float pDensity;
    Vec3f normal = sampleSphereSurfaceUniformly(rng, pDensity);
    Vec3f wo = sampleHemisphereSurfaceUniformly(rng, normal, pDensity);
    Vec3f tangent = anyOrthonormal(normal);
    Vec3f bitangent = cross(normal, tangent);
    Vec3f wi = sampleGGXVNDFGlobal(rng, normal, alpha, alpha, tangent, bitangent, wo, pDensity);
    EXPECT_TRUE(pDensity >= 0.0f);
    EXPECT_NEARREL(norm(wi), 1.0f, 1e-4f);
    float pDensity2 = sampleGGXVNDFGlobalDensity(normal, alpha, alpha, tangent, bitangent, wo, wi);
    EXPECT_NEARREL(pDensity, pDensity2, 1e-2f );
  }
}
TEST(brdf, sampleGGXVNDFGlobal2)
{
  RNG rng(0);
  
  float alpha = 1.0f;
  float pDensity;
  Vec3f normal = sampleSphereSurfaceUniformly(rng, pDensity);
  // we can only use this as a sampling distribution if we choose wo = normal, otherwise not the whole hemisphere is properly samples....
  Vec3f wo = normal;

  float sum = 0.0f;
  float n = 5000000;
  for (int i = 0; i < n; i++)
  {
    Vec3f tangent = anyOrthonormal(normal);
    Vec3f bitangent = cross(normal, tangent);
    Vec3f wi = sampleGGXVNDFGlobal(rng, normal, alpha, alpha, tangent, bitangent, wo, pDensity);
    EXPECT_TRUE(pDensity >= 0.0f);
    sum += std::max(0.0f, dot(normal, wi)) / pDensity;
  }
  float estimate = sum / n;
  EXPECT_NEARREL(estimate, float(Pi), 1e-2f );
}

TEST(brdf, refraction)
{
  Vec3f normal = normalize(Vec3f(0.1f, 0.2f, 1.0f));
  Vec3f wi = normalize(Vec3f(0.15f, 0.05f, 1.0f));
  Vec3f wo, wi2;
  //checking going in and out gives the same direction
  computeRefractionDirection(wi, normal, 1.45f, 1.0f, wo);
  computeRefractionDirection(wo, -normal, 1.0f, 1.45f, wi2);
  
  EXPECT_NEARREL(wi[0], wi2[0], 1e-5f);
  EXPECT_NEARREL(wi[1], wi2[1], 1e-5f);
  EXPECT_NEARREL(wi[2], wi2[2], 1e-5f);

  //check symmetry
  float cos_i = 0.3f;
  float cos_t = 0.6f;
  float F1 = fresnelDielectric(cos_i, cos_t, 1.0f, 1.4f);
  float F2 = fresnelDielectric(cos_t, cos_i, 1.4f, 1.0f);
  EXPECT_NEARREL(F1, F2, 1e-5f);

}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
