#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Random.h>

namespace Hasty
{

Vec2f sampleDiskUniformly(RNG& rng, float* pDensity);
Vec3f sampleSphereSurfaceUniformly(RNG& rng, float* pDensity);

Vec3f sampleHemisphereSurfaceUniformly(RNG& rng, const Vec3f& normal, float* pDensity);

Vec3f sampleHemisphereCosImportance(RNG& rng, const Vec3f& normal, float* pDensity);
float evaluateHemisphereCosImportancePDF(const Vec3f& normal, const Vec3f& direction);

Vec3f sampleCosineLobe(RNG& rng, const Vec3f& lobeDirection, float exponent, float* pDensity);
float evaluateCosineLobePDF(const Vec3f& lobeDirection, float exponent, const Vec3f& direction);

Vec3f sampleTriangleUniformly(RNG& rng);

}
