#pragma once

#include <Hasty/Core.h>
#include <Hasty/VMath.h>
#include <Hasty/Random.h>

namespace Hasty
{

Vec2f sampleDiskUniformly(HASTY_INOUT(RNG) rng, HASTY_OUT(float) pDensity);
Vec2f sampleCircularNGonUniformly(HASTY_INOUT(RNG) rng, int numBlades, float bladeRotation, HASTY_OUT(float) pDensity);

Vec3f sampleSphereSurfaceUniformly(HASTY_INOUT(RNG) rng, HASTY_OUT(float) pDensity);

Vec3f sampleHemisphereSurfaceUniformly(HASTY_INOUT(RNG) rng, const Vec3f& normal, HASTY_OUT(float) pDensity);

Vec3f sampleHemisphereCosImportance(HASTY_INOUT(RNG) rng, const Vec3f& normal, HASTY_OUT(float) pDensity);
float evaluateHemisphereCosImportancePDF(const Vec3f& normal, const Vec3f& direction);

Vec3f sampleCosineLobe(HASTY_INOUT(RNG) rng, const Vec3f& lobeDirection, float exponent, HASTY_OUT(float) pDensity);
float evaluateCosineLobePDF(const Vec3f& lobeDirection, float exponent, const Vec3f& direction);

Vec3f sampleTriangleUniformly(HASTY_INOUT(RNG) rng);

}
