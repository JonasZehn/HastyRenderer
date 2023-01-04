#include <Hasty/BRDF.h>

#include <Hasty/Scene.h>
#include <Hasty/PathTracing.h>

namespace Hasty
{

#define HASTY_POWCI(v, constantIntegerExponent) powci<constantIntegerExponent>(v);

bool computeRefractionDirection(const Vec3f& wi, const Vec3f& normal, float indexOfRefraction_i, float indexOfRefraction_t, HASTY_OUT(Vec3f) wt)
{
  // t stands for transmittance
  //https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#sec:specular-transmit
  float relIOR = indexOfRefraction_i / indexOfRefraction_t;
  float cos_i = dot(wi, normal);
  assert(cos_i >= 0.0f);
  //float side = cos_i >= 0.0f ? 1.0f : -1.0f;
  float sin_iSq = std::max(0.f, 1.f - cos_i * cos_i);
  float sin_tSq = (relIOR * relIOR) * sin_iSq;
  if(sin_tSq >= 1.0f)
  {
    // in this case we get total internal reflection and we can't compute cos_t (going complex)
    return false;
  }
  float cos_t = std::sqrt(1.0f - sin_tSq);
  wt = (-relIOR) * wi + (relIOR * cos_i - cos_t) * normal;
  wt = normalize(wt);
  return true;
}
bool computeRefractionCosT(float cos_i, float indexOfRefraction_i, float indexOfRefraction_t, HASTY_OUT(float) cos_t)
{
  assert(cos_i >= 0.0f);

  // t stands for transmittance
  //https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#sec:specular-transmit
  float relIOR = indexOfRefraction_i / indexOfRefraction_t;

  //float side = cos_i >= 0.0f ? 1.0f : -1.0f;
  float sin_iSq = std::max(0.f, 1.f - cos_i * cos_i);
  float sin_tSq = (relIOR * relIOR) * sin_iSq;
  if(sin_tSq >= 1.0f)
  {
    // in this case we get total internal reflection and we can't compute cos_t (going complex)
    return false;
  }
  cos_t = std::sqrt(1.0f - sin_tSq);
  return true;
}

// normal points in the same direction as wi
Vec3f computeRefractionDirectionFromAngles(const Vec3f& wi, const Vec3f& normal, float IORi_over_IORt, float cos_i, float cos_t)
{
  assert(dot(wi, normal) >= 0.0f);
  assert(cos_i >= 0.0f);
  assert(cos_t >= 0.0f);

  Vec3f wt = (-IORi_over_IORt) * wi + (IORi_over_IORt * cos_i - cos_t) * normal;
  wt = normalize(wt);
  return wt;
}

//https://psgraphics.blogspot.com/2016/12/bug-in-my-schlick-code.html?m=0
inline float fresnelSchlick(float cos_i, float indexOfRefraction_i, float indexOfRefraction_t)
{
  //https://en.wikipedia.org/wiki/Schlick%27s_approximation
  float R0 = square((indexOfRefraction_i - indexOfRefraction_t) / (indexOfRefraction_i + indexOfRefraction_t));
  float cosT = cos_i;
  return R0 + (1.0f - R0) * HASTY_POWCI(1.0f - cosT, 5);
}

// https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
// https://web.cse.ohio-state.edu/~parent.1/classes/782/Lectures/05_Reflectance.pdf
float fresnelDielectric(float cos_i, float cos_t, float indexOfRefraction_i, float indexOfRefraction_t)
{
  assert(cos_i >= 0.0f);
  assert(cos_t >= 0.0f);
  float rp = (indexOfRefraction_t * cos_i - indexOfRefraction_i * cos_t) / (indexOfRefraction_t * cos_i + indexOfRefraction_i * cos_t);
  float ro = (indexOfRefraction_i * cos_i - indexOfRefraction_t * cos_t) / (indexOfRefraction_i * cos_i + indexOfRefraction_t * cos_t);
  return 0.5f * (rp * rp + ro * ro);
}

// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
// https://schuttejoe.github.io/post/ggximportancesamplingpart1/
Vec3f sampleDGGX(HASTY_INOUT(RNG) rng, const Vec3f& normal, float alpha, const Vec3f& dir1, HASTY_OUT(float) pDensity)
{
  if(dot(dir1, normal) < 0.0f)
  {
    std::cout << " warning sampleDGGX from the inside" << '\n';
    pDensity = 1.0f;
    return -normal;
  }
  // theta_h ~ alpha*alpha * cos(theta_h) sin(theta_h) / (float(Pi) * (q * q)), q = (cos(theta_h) ^2 ) * (alpha * alpha - 1.0f) + 1.0f,   q(0) = a^2, q(\pi/2) = 1
  // F(\theta_h) = \int_0^{\theta_h} 1/C alpha*alpha * cos(\theta_h') / (float(Pi) * (q * q)) d\theta_h' =  [ 1/(Pi C) a^2 / ( 2  (a^2 - 1) q(\theta') )  ]_0^\theta_h = 1/(Pi C) a^2 / ( 2  (a^2 - 1) q(\theta_h) ) - 1/(Pi C) a^2 / ( 2  (a^2 - 1) q(0) )
  //             = 1/(Pi C) a^2 / ( 2  (a^2 - 1) q(\theta_h) ) - 1/(Pi C  2  (a^2 - 1)  ) = 1 / (Pi C 2  q(\theta_h) )  
  // F(\pi/2) = 1 = 1 / (Pi C 2) => C =  1 / (pi 2  ) 
  // F(\theta_h) = 1 / q(\theta_h) = u = 1 / ((nh * nh) * (alpha * alpha - 1.0f) + 1.0f) , 1 = (nh * nh) * (alpha * alpha - 1.0f) u + u
  //            (1 - u)/(u (alpha^2 - 1) ) = cos(\theta_h)^2
  Vec3f h;
  do
  {
    float xi1 = uniform01f(rng);
    float xi2 = uniform01f(rng);

    float phi = 2.0f * float(Pi) * xi2;
    float costheta = std::sqrt((1.0f - xi1) / (xi1 * (alpha * alpha - 1.0f) + 1.0f)); // using double here on purpose due to numerical problems
    float sintheta = std::sqrt(1.0f - costheta * costheta);
    float cosphi = std::cos(phi);
    float sinphi = std::sin(phi); // you cannot use sqrt(1 - cosphi*cosphi) here you are gonna loose the sign
    Vec3f rv = Vec3f(sintheta * cosphi, sintheta * sinphi, costheta);

    RotationBetweenTwoVectors<float, Vec3f> rotation(Vec3f(0.0f, 0.0f, 1.0f), normal);
    h = applyRotation(rotation, rv);
    h = normalize(h);
  } while(dot(h, normal) <= 0.0f);

  // p(dir2) = p( h  | dir1) / abs(det(jac_h(transformation)))
  // dir2 =  2 * dot(h, dir1) * h - dir1
  //  we can just use http://www.matrixcalculus.org/ for the jacobian, sth like: 2 *h * (h'* y) - y, giving  2 h y' + 2 h' y * Identity, determinant is 4 h' y, i guess
  Vec3f dir2 = reflectAcross(dir1, h);
  dir2 = normalize(dir2);

  float nh = dot(normal, h);
  float q = (nh * nh) * (alpha * alpha - 1.0f) + 1.0f;
  float pDensityH = alpha * alpha * nh / (float(Pi) * (q * q));
  pDensity = pDensityH / std::abs(4.0f * dot(dir1, h));
  return dir2;
}
float sampleDGGXDensity(const Vec3f& normal, float alpha, const Vec3f& dir1, const Vec3f& dir2)
{
  Vec3f h = normalize(dir1 + dir2); // this is not the "correct" h, the sign may be wrong wrt sampleDGGX, which is why we just use abs on the next line
  float nh = std::abs(dot(normal, h));
  float q = (nh * nh) * (alpha * alpha - 1.0f) + 1.0f;
  float pDensityH = alpha * alpha * nh / (float(Pi) * (q * q));
  return pDensityH / (1e-7f + std::abs(4.0f * dot(dir1, h)));
}
float DGGX(float nh, float alpha)
{
  float q = (nh * nh) * (alpha * alpha - 1.0f) + 1.0f;
  float D = alpha * alpha / (1e-7f + float(Pi) * (q * q));
  return D;
}
float SmithIsotropicGGXShadowing(float alpha, float vsq, float vn)
{
  float lambda = (std::sqrt(1.0f + alpha * alpha * (vsq / (vn * vn) - 1.0f)) - 1.0f) * 0.5f;
  return 1.0f / (1.0f + lambda);
}

float DGGX(float nh, float th, float bh, float alpha_t, float alpha_b)
{
  float D = float(Hasty::InvPi) / (alpha_t * alpha_b * Hasty::square(Hasty::square(th / alpha_t) + Hasty::square(bh / alpha_b) + Hasty::square(nh)));
  return D;
}
// "glass ground unknown"
float smithGGX(float nv, float tv, float bv, float alpha_t, float alpha_b)
{
  float lambda = 0.5f * (std::sqrt(1.0f + (square(alpha_t * tv) + square(alpha_b * bv)) / square(nv)) - 1.0f);
  return 1.0f / (1.0f + lambda);
}
// see "Sampling the GGX Distribution of Visible Normals" Eric Heitz for an explanation
// and https://simonstechblog.blogspot.com/2020/01/note-on-sampling-ggx-distribution-of.html
// and https://schuttejoe.github.io/post/ggximportancesamplingpart2/
Vec3f sampleGGXVNDF(Vec3f wOut, float alphaX, float alphaY, float xi1, float xi2)
{
  Vec3f Vh = normalize(Vec3f(alphaX * wOut[0], alphaY * wOut[1], wOut[2]));

  // compute frame
  Vec3f X = Vec3f(1.0f, 0.0f, 0.0f);
  Vec3f Z = Vec3f(0.f, 0.f, 1.0f);
  Vec3f T1 = cross(Vh, Z);
  if(normSq(T1) < 1e-12f) T1 = X;
  else T1 = normalize(T1);
  Vec3f T2 = cross(T1, Vh);

  float r = std::sqrt(xi1);
  float a = 1.0f / (1.0f + Vh[2]);
  float phi = float(Pi) * ((xi2 < a) ? (xi2 / a) : (1.0f + (xi2 - a) / (1.0f - a)));
  float t1 = r * std::cos(phi);
  float t2 = r * std::sin(phi) * (xi2 < a ? 1.0f : Vh[2]);
  float t3 = std::sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2));
  Vec3f Nh = t1 * T1 + t2 * T2 + t3 * Vh;
  Vec3f Ne = normalize(Vec3f(alphaX * Nh[0], alphaY * Nh[1], std::max(0.0f, Nh[2])));

  return Ne;
}
Vec3f sampleGGXVNDFGlobal(HASTY_INOUT(RNG) rng, const Vec3f& normal, float alpha_t, float alpha_b, const Vec3f& tangent, const Vec3f& bitangent, const Vec3f& dir1, HASTY_OUT(float) pDensity)
{
  if(dot(dir1, normal) <= 0.0f)
  {
    std::cout << " warning sampleGGXVNDFGlobal from the inside" << '\n';
    pDensity = 0.0f;
    return -normal;
  }

  Vec3f NeGlobal;
  do
  {
    Vec3f Ve = Vec3f(dot(tangent, dir1), dot(bitangent, dir1), dot(normal, dir1));
    Vec3f Ne = sampleGGXVNDF(Ve, alpha_t, alpha_b, uniform01f(rng), uniform01f(rng));

    NeGlobal = tangent * Ne[0] + bitangent * Ne[1] + normal * Ne[2];
    NeGlobal = normalize(NeGlobal);
  } while(dot(NeGlobal, dir1) <= 0.0f);

  Vec3f dir2 = reflectAcross(dir1, NeGlobal);
  dir2 = normalize(dir2);
  // V = dir1, N = H
  float G1 = smithGGX(dot(normal, dir1), dot(tangent, dir1), dot(bitangent, dir1), alpha_t, alpha_b);
  float D_ggx = DGGX(dot(normal, NeGlobal), dot(tangent, NeGlobal), dot(bitangent, NeGlobal), alpha_t, alpha_b);
  float pDensityH = G1 * dot(dir1, NeGlobal) * D_ggx / dot(dir1, normal);
  pDensity = pDensityH / (1e-7f + 4.0f * std::abs(dot(dir1, NeGlobal)));
  return dir2;
}
float sampleGGXVNDFGlobalDensity(const Vec3f& normal, float alpha_t, float alpha_b, const Vec3f& tangent, const Vec3f& bitangent, const Vec3f& dir1, const Vec3f& dir2)
{
  if(dot(dir1, normal) <= 0.0f)
  {
    std::cout << " warning sampleGGXVNDFGlobalDensity from the inside" << '\n';
    return 0.0f;
  }
  Vec3f NeGlobal = normalize(dir1 + dir2);
  float G1 = smithGGX(dot(normal, dir1), dot(tangent, dir1), dot(bitangent, dir1), alpha_t, alpha_b);
  float D_ggx = DGGX(dot(normal, NeGlobal), dot(tangent, NeGlobal), dot(bitangent, NeGlobal), alpha_t, alpha_b);
  float pDensityH = G1 * std::max(0.0f, dot(dir1, NeGlobal)) * D_ggx / dot(dir1, normal);
  return pDensityH / (1e-7f + 4.0f * std::abs(dot(dir1, NeGlobal)));
}


inline float signSide(float dotNW)
{
  return dotNW < 0.0f ? -1.0f : 1.0f;
}

inline bool sideOutside(float dotNW)
{
  return dotNW >= 0.0f;
}

inline float computeNormalScale(const Vec3f& wi, const Vec3f& normalShading, const Vec3f& normalGeometric)
{
  return std::abs(dot(wi, normalShading)) / (1e-7f + std::abs(dot(wi, normalGeometric)));
}


// Goal : mix(mix(DielectricBRDF, GlassBRDF,  transmission), metallicBRDF, metallic)
//   where DielectricBRDF, GlassBRDF and metallicBRDF is a fully working BRDF, and the grouping like this is cause glass is a dielectric...
// but this can be simplified because we use a simpler model where the specular lobe of the metallic, the dielectric and the "glass"
// mix(mix(DielectricBRDFDiffuse, GlassBRDFDiffuse,  transmission), metallicBRDFDiffuse, metallic)
//  = mix(mix(DielectricBRDFDiffuse, 0,  transmission), 0, metallic)
//  = DielectricBRDFDiffuse * (1.0 - transmission) * (1.0 - metallic)
//  = Lambertian * (1.0f - 0.5f * specularHit) * (1.0 - transmissionHit) * (1.0 - metallicHit)

float PrincipledBRDF::computeAlpha(float roughness)
{
  //we have alpha min to not have dirac distribution as special case
  float alpha = roughness * roughness;
  if(alpha < 1e-2f && alpha != 0.0f) // we handle alpha == 0.0f specifically
  {
    //std::cout << " warning: roughness very small, clamping due to instabilities " << std::endl;
    alpha = 1e-2f;
  }
  return alpha;
}
PrincipledBRDF::ProbabilityResult PrincipledBRDF::computeProbability(float metallicHit, float specularHit, float transmissionHit, float cos_o, float indexOfRefraction_o, float indexOfRefraction_refr_o, ShaderEvalFlag evalFlag)
{
  assert(cos_o >= 0.0f);
  PrincipledBRDF::ProbabilityResult result;
  result.refractionPossible = computeRefractionCosT(cos_o, indexOfRefraction_o, indexOfRefraction_refr_o, result.cosT);
  if(result.refractionPossible)
  {
    result.FDaccurate = fresnelDielectric(cos_o, result.cosT, indexOfRefraction_o, indexOfRefraction_refr_o); // goes from 0 to 1 for cos_i going from 0 to 90 degrees, at higher degrees we get more reflection
  }
  else
  {
    result.FDaccurate = 1.0f;
  }

  float mDiffuseStrategy = (1.0f - transmissionHit) * (1.0f - metallicHit) * (1.0f - 0.5f * specularHit); // we want this to be zero when metallic and one when specular = 0
  float sTransmission = result.FDaccurate;
  float sMetallic = 1.0f;
  float mSpecularStrategy = mix(mix(0.5f * specularHit, sTransmission, transmissionHit), sMetallic, metallicHit);
  float mRefractiveStrategy = (1.0 - result.FDaccurate) * transmissionHit * (1.0 - metallicHit);

  if(!has(evalFlag, ShaderEvalFlag::DIFFUSE)) mDiffuseStrategy = 0.0f;
  if(!has(evalFlag, ShaderEvalFlag::CONCENTRATED))
  {
    mSpecularStrategy = 0.0f;
    mRefractiveStrategy = 0.0f;
  }

  float mSum = mDiffuseStrategy + mSpecularStrategy + mRefractiveStrategy;
  result.noStrategy = mSum == 0.0f;
  if(result.noStrategy)
  {
    result.pDiffuseStrategy = 0.0f;
    result.pSpecularStrategy = 0.0f;
    result.pRefractiveStrategy = 0.0f;
  }
  else
  {
    result.pDiffuseStrategy = mDiffuseStrategy / mSum;
    result.pSpecularStrategy = mSpecularStrategy / mSum;
    result.pRefractiveStrategy = mRefractiveStrategy / mSum;
  }

  return result;
}
void PrincipledBRDF::computeAnisotropyParameters(const SurfaceInteraction& interaction, const Vec3f& normalShading, float alpha, float anisotropyHit, HASTY_OUT(float) alpha_t, HASTY_OUT(float) alpha_b, HASTY_OUT(Vec3f) tangent, HASTY_OUT(Vec3f) bitangent)
{
  float aspect = std::sqrt(1.0f - 0.9f * anisotropyHit);
  alpha_t = alpha / aspect;
  alpha_b = alpha * aspect;
  RotationBetweenTwoVectors<float, Vec3f> rotation(interaction.normalGeometric, normalShading);
  tangent = applyRotation(rotation, interaction.tangent);
  bitangent = applyRotation(rotation, interaction.bitangent);
}

// this function assumes that dot(Ns, wo) >= 0
Vec3f PrincipledBRDF::evaluateDiffuse(float dotNsWi, const Vec3f& albedoHit, float normalScale, float metallicHit, float specularHit, float transmissionHit)
{
  if(dotNsWi < 0.0f)
  {
    return Vec3f::Zero();
  }
  return albedoHit * (float(InvPi) * normalScale * (1.0f - 0.5f * specularHit) * (1.0 - transmissionHit) * (1.0 - metallicHit));
}
// this function assumes that dot(Ns, wo) >= 0
Vec3f PrincipledBRDF::evaluateConcentratedWithoutTransmission(
  float dotNsWi,
  const Vec3f& albedoHit, float metallicHit, float specularHit,
  float roughnessHit, float anisotropyHit, float transmissionHit,
  const Vec3f& wo, const Vec3f& wi,
  float alpha,
  float alpha_t, float alpha_b,
  const Vec3f& tangent, const Vec3f& bitangent,
  const Vec3f& normalShading,
  float normalScale,
  bool forceHitSpecular,
  const Vec3f& FDaccurate)
{
  if(dotNsWi < 0.0f) return Vec3f::Zero();

  Vec3f h = normalize(wo + wi);
  float lh = std::abs(dot(wi, h));

  //Pseudo fresnel:
  // https://www.youtube.com/watch?v=kEcDbl7eS0w basically F0 is directly the color one desires in the middle, and it's actually more principled to use F0 + SChlick than using the full fresnel equations because we don't do proper spectral rendering
  float cosNih5 = HASTY_POWCI(1.0f - lh, 5);
  // FH * D * G / (4 * nv * nl) with:
  //Vec3f F0 = mix(Vec3f::Ones(), albedoHit, metallicHit); // and we also have a specular part for non metalics, in that case it's kinda white, wheras for a metallic object the light gets "colored"
  //Vec3f FH = F0 + (Vec3f::Ones() - F0) * cosNih5; // here if we are at a grazing angle, we  remove some F0, and add some "achromatic" reflectance (Vec3f::ones)
  //Vec3f FH = F0 + Vec3f::Ones() * cosNih5 - F0 * cosNih5;
  //Vec3f FH = mix(Vec3f::Ones(), albedoHit, metallicHit) * (1.0f - cosNih5) + Vec3f::Ones() * cosNih5;
  //Vec3f FH = mix(Vec3f::Ones() * (1.0f - cosNih5), albedoHit * (1.0f - cosi5), metallicHit) + Vec3f::Ones() * cosNih5;
  //Vec3f FH = mix(Vec3f::Ones() * (1.0f - cosNih5) + Vec3f::Ones() * cosNih5, albedoHit * (1.0f - cosNih5) + Vec3f::Ones() * cosi5, metallicHit);
  //Vec3f FH = mix(Vec3f::Ones(), albedoHit * (1.0f - cosNih5) + Vec3f::Ones() * cosNih5, metallicHit);

  float nh = dot(normalShading, h);

  float nv = std::abs(dot(normalShading, wo));
  float nl = std::abs(dot(normalShading, wi));
  float Go = smithGGX(nv, dot(tangent, wo), dot(bitangent, wo), alpha_t, alpha_b);
  float Gi = smithGGX(nl, dot(tangent, wi), dot(bitangent, wi), alpha_t, alpha_b);
  float G = Go * Gi;

  float D_ggx;

  if(alpha == 0.0f)
  {
    // D_GGX explodes for alpha= 0 and nh = 1
    // we can find this by using importance sample formula and then let alpha go to zero?, G is always 1 in that case
    // => \int F D G/ (4 nv nl) dl ~= 1/N \sum_i  F D Gi Go/ (4 nv nl_i  p(l_i) )
    //           ~= 1/N \sum_i  F / nv ~=   F / nv for alpha = 0
    D_ggx = forceHitSpecular ? 4.0f * nl : 0.0f; // or 4.0f * nv? nv and nl should be the same for a fully reflective hit
  }
  else
  {
    D_ggx = DGGX(nh, dot(tangent, h), dot(bitangent, h), alpha_t, alpha_b);
  }

  float specularFromDielectric = 0.5f * specularHit;
  Vec3f specFromTransmission = Vec3f::Zero();
  Vec3f specularMetallic = albedoHit * (1.0f - cosNih5) + Vec3f::Ones() * cosNih5;

  Vec3f resultTransmission = Vec3f::Zero();
  if(transmissionHit > 0.0f)
  {
    if(alpha != 0.0f) throw std::runtime_error("not implemented");
    specFromTransmission = forceHitSpecular ? FDaccurate : Vec3f::Zero();
  }
  Vec3f resultSpecular = mix(mix(Vec3f::Fill(specularFromDielectric), specFromTransmission, transmissionHit), specularMetallic, metallicHit) * (D_ggx * G * normalScale / std::abs(1e-7f + 4.0f * nv * nl));
  Vec3f result = resultSpecular;
  return result;
}

//wavelength \in [-1, 0, 1, 2] where -1 means the same index of refraction for all wavelengths
SampleResult PrincipledBRDF::sample(
  HASTY_INOUT(RNG) rng, const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& normalShading,
  const Vec3f& albedoHit,
  float metallicHit, float specularHit, float roughnessHit, float anisotropyHit, float transmissionHit,
  Wavelength wavelength, float indexOfRefractionOutside, float indexOfRefractionMat, bool adjoint, ShaderEvalFlag evalFlag)
{
  SampleResult result;

  assert(evalFlag != ShaderEvalFlag::NONE); // no distribution to sample
  assert(dot(normalShading, wo) >= 0.0f);

  float dotNgWo = dot(interaction.normalGeometric, wo);
  float dotNsWo = dot(normalShading, wo);

  float alpha = PrincipledBRDF::computeAlpha(roughnessHit);

  result.outside = sideOutside(dotNgWo);
  float indexOfRefraction_o = result.outside ? indexOfRefractionOutside : indexOfRefractionMat;
  float indexOfRefraction_refr_o = result.outside ? indexOfRefractionMat : indexOfRefractionOutside;

  PrincipledBRDF::ProbabilityResult probabilityResult = PrincipledBRDF::computeProbability(metallicHit, specularHit, transmissionHit, dotNsWo, indexOfRefraction_o, indexOfRefraction_refr_o, evalFlag);

  if(probabilityResult.noStrategy)
  {
    result.direction = Vec3f(1.0f, 0.0f, 0.0f);
    result.throughputDiffuse = Vec3f::Zero();
    result.throughputConcentrated = Vec3f::Zero();
    result.pdfOmega = 0.0f;
    return result;
  }

  // we just put the same Fresnel term in for all wavelengths and then mask the output:
  // we may need to change a lot fo rough glass material...
  Vec3f FDaccurateVector = Vec3f::Fill(probabilityResult.FDaccurate);
  Vec3f wavelengthMask = Vec3f::Ones();
  if(wavelength != -1)
  {
    wavelengthMask = Vec3f::Zero();
    wavelengthMask[wavelength] = 1.0f;
  }

  float alpha_t, alpha_b;
  Vec3f tangent, bitangent;
  PrincipledBRDF::computeAnisotropyParameters(interaction, normalShading, alpha, anisotropyHit, alpha_t, alpha_b, tangent, bitangent);

  // pSpecStr = proability to sample using Specular Strategy
  // p_diff = probability density of diffuse sampling scheme
  // p_spec = probability density of specular sampling scheme
  //F = f L cos(omega) / p(omega)  , omega ~ uniform
  //E[F] = \int f L cos(omega) / p(omega) * p(omega) domega = \int f L cos(omega) domega
  //F' = {
  //      f L cos(omega) / (pSpecStr p_spec(omega) + (1 - pSpecStr) p_diff(omega)) , omega ~ p(spec)     xi < pSpecStr
  //      f L cos(omega) / (pSpecStr p_spec(omega) + (1 - pSpecStr) p_diff(omega)), omega ~ p(diffuse) xi >= pSpecStr
  //     }
  //E[F'] = pSpecStr \int f L cos(omega) / (pSpecStr p_spec(omega) + (1 - pSpecStr) p_diff(omega))  p_spec(omega) domega + (1 - pSpecStr) \int f L cos(omega) / (pSpecStr p_spec(omega) + (1 - pSpecStr) p_diff(omega)) p_diff(omega) domega
  //       = \int f L cos(omega)  (pSpecStr  p_spec(omega) + (1 - pSpecStr)p_diff(omega)) / (pSpecStr p_spec(omega) + (1 - pSpecStr) p_diff(omega))) domega
  //       = \int f L cos(omega) domega = E[F]

  MaterialEvalResult evalResult = MaterialEvalResult(Vec3f::Zero(), Vec3f::Zero(), normalShading);

  // using defines here instead of enums to support glsl
#define PBRDF_STRATEGY_DIFFUSE 1u
#define PBRDF_STRATEGY_SPECULAR 2u
#define PBRDF_STRATEGY_REFRACTIVE 3u

  float xi = uniform01f(rng); // xi \in [0, 1)
  uint32_t strategy = xi < probabilityResult.pRefractiveStrategy ? PBRDF_STRATEGY_REFRACTIVE : (xi < (probabilityResult.pRefractiveStrategy + probabilityResult.pSpecularStrategy) ? PBRDF_STRATEGY_SPECULAR : PBRDF_STRATEGY_DIFFUSE);
  if(strategy == PBRDF_STRATEGY_SPECULAR)
  {

    if(alpha == 0.0f) // delta distribution; need to keep wierd fake measure
    {
      //F' = {
      //      (f_specular L cos(omegaSpec)) / pSpecStr, omega = omegaSpec     xi < pSpecStr
      //      f_diffuse L cos(omega) / ((1 - pSpecStr) p_diff), omega ~ p_diff xi >= pSpecStr
      //     }
      //E[F'] = pSpecStr f_specular(omega) L(omega) cos(omegaSpec) / pSpecStr + (1 - pSpecStr) \int f_diffuse L cos(omega) / ((1 - pSpecStr) pDiffuse) pDiffuse(omega) domega
      //       = L f_specular(omega_specular) cos(omega_specular) + \int L cos(omega) f_diffuse(omega) domega
      //       = \int  L cos(omega) f_specular dirac_specular(omega) domega + \int L cos(omega) f_diffuse domega
      //       = \int  L cos(omega) (f_specular dirac_specular(omega)  +  f_diffuse) domega
      result.direction = reflectAcross(wo, normalShading);
      result.pdfOmega = 1e12f; // this is the probability for MIS , we could also tell it directly that it's a dirac delta but this should work to working precision
      const Vec3f& wi = result.direction;
      float dotNsWi = dot(normalShading, wi);
      float normalScale = computeNormalScale(wi, normalShading, interaction.normalGeometric);
      evalResult.fDiffuse = Vec3f::Zero();
      evalResult.fConcentrated = 1e12f / probabilityResult.pSpecularStrategy * PrincipledBRDF::evaluateConcentratedWithoutTransmission(
        dotNsWi,
        albedoHit, metallicHit, specularHit,
        roughnessHit, anisotropyHit, transmissionHit,
        wo, result.direction,
        alpha,
        alpha_t, alpha_b,
        tangent, bitangent,
        normalShading, normalScale, true, FDaccurateVector);
    }
    else
    {
      int k = 3;
      RNG backup = rng;
      rng = backup;
      float pdfSpec;
      result.direction = sampleGGXVNDFGlobal(rng, normalShading, alpha_t, alpha_b, tangent, bitangent, wo, pdfSpec);
      float pdfDiffuse = evaluateHemisphereCosImportancePDF(normalShading, result.direction);

      result.pdfOmega = probabilityResult.pSpecularStrategy * pdfSpec + probabilityResult.pDiffuseStrategy * pdfDiffuse;
      const Vec3f& wi = result.direction;
      float dotNsWi = dot(normalShading, wi);
      float normalScale = computeNormalScale(wi, normalShading, interaction.normalGeometric);
      evalResult.fDiffuse = PrincipledBRDF::evaluateDiffuse(dotNsWi, albedoHit, normalScale, metallicHit, specularHit, transmissionHit);
      evalResult.fConcentrated = PrincipledBRDF::evaluateConcentratedWithoutTransmission(
        dotNsWi,
        albedoHit, metallicHit, specularHit,
        roughnessHit, anisotropyHit, transmissionHit,
        wo, result.direction,
        alpha,
        alpha_t, alpha_b,
        tangent, bitangent,
        normalShading, normalScale, false, FDaccurateVector);
    }
  }
  else if(strategy == PBRDF_STRATEGY_DIFFUSE)
  {
    float pdfDiffuse;
    result.direction = sampleHemisphereCosImportance(rng, normalShading, pdfDiffuse);
    const Vec3f& wi = result.direction;
    float dotNsWi = dot(normalShading, wi);
    float normalScale = computeNormalScale(wi, normalShading, interaction.normalGeometric);

    if(alpha == 0.0f) // delta distribution; need to keep wierd fake measure
    {
      result.pdfOmega = probabilityResult.pDiffuseStrategy * pdfDiffuse;
      evalResult.fDiffuse = PrincipledBRDF::evaluateDiffuse(dotNsWi, albedoHit, normalScale, metallicHit, specularHit, transmissionHit);
      evalResult.fConcentrated = Vec3f::Zero();
    }
    else
    {
      float pdfSpec = sampleGGXVNDFGlobalDensity(normalShading, alpha_t, alpha_b, tangent, bitangent, wo, result.direction);
      result.pdfOmega = probabilityResult.pSpecularStrategy * pdfSpec + (1.0f - probabilityResult.pSpecularStrategy) * pdfDiffuse;
      evalResult.fDiffuse = PrincipledBRDF::evaluateDiffuse(dotNsWi, albedoHit, normalScale, metallicHit, specularHit, transmissionHit);
      evalResult.fConcentrated = PrincipledBRDF::evaluateConcentratedWithoutTransmission(
        dotNsWi,
        albedoHit, metallicHit, specularHit,
        roughnessHit, anisotropyHit, transmissionHit,
        wo, result.direction,
        alpha,
        alpha_t, alpha_b,
        tangent, bitangent,
        normalShading, normalScale, false, FDaccurateVector);
    }
  }
  else
  {
    // refraction: switch outside
    result.outside = !result.outside;
    result.pdfOmega = 1e12f;
    evalResult.fDiffuse = Vec3f::Zero();
    if(!probabilityResult.refractionPossible)
    {
      evalResult.fConcentrated = Vec3f::Zero();
      result.direction = Vec3f(1.0f, 0.0f, 0.0f);
    }
    else
    {
      float iorO_over_iorRo = indexOfRefraction_o / indexOfRefraction_refr_o;
      Vec3f wi = computeRefractionDirectionFromAngles(wo, normalShading, iorO_over_iorRo, dotNsWo, probabilityResult.cosT);
      result.direction = wi;
      float scale = adjoint ? 1.0f : square(iorO_over_iorRo); // See "Non-symmetric Scattering in Light Transport Algorithms" the eq. after eq. 10
      float fc = result.pdfOmega / probabilityResult.pRefractiveStrategy * scale * (1.0f - probabilityResult.FDaccurate) / std::abs(dot(result.direction, interaction.normalGeometric));
      evalResult.fConcentrated = fc * wavelengthMask;
    }
  }

  float m = std::abs(dot(result.direction, interaction.normalGeometric)) / result.pdfOmega;
  result.throughputDiffuse = evalResult.fDiffuse * m;
  result.throughputConcentrated = evalResult.fConcentrated * m;

  assertUnitLength(result.direction);
  assertFinite(result.throughputDiffuse);
  assertFinite(result.throughputConcentrated);
  assertFinite(result.pdfOmega);
  return result;
}

PrincipledBRDF::PrincipledBRDF(std::unique_ptr<ITextureMap3f> albedo, std::unique_ptr<ITextureMap1f> roughness, std::unique_ptr<ITextureMap1f> metallic, float specular, float indexOfRefraction, std::unique_ptr<ITextureMap3f> normalMap, float anisotropy, float transmission, bool varyIOR)
  :albedo(std::move(albedo)), roughness(std::move(roughness)), metallic(std::move(metallic)), specular(specular), indexOfRefraction(indexOfRefraction), normalMap(std::move(normalMap)), anisotropy(anisotropy), transmission(transmission), varyIOR(varyIOR)
{
}
PrincipledBRDF::~PrincipledBRDF()
{

}
float PrincipledBRDF::getIndexOfRefraction(int wavelength) const
{
  if(!varyIOR)
  {
    return indexOfRefraction;
  }
  assert(wavelength != -1);
  switch(wavelength)
  {
  case 0: return indexOfRefraction * (1.43f / 1.45f);
  case 1: return indexOfRefraction * (1.45f / 1.45f);
  case 2: return indexOfRefraction * (1.47f / 1.45f);
  default: return std::numeric_limits<float>::signaling_NaN();
  }
}
Vec3f PrincipledBRDF::getShadingNormal(const SurfaceInteraction& interaction, const Vec3f& wo, float dotNgWo)
{
  float sideNgWo = signSide(dotNgWo);
  Vec3f normalShading;
  // first we make sure we have a normal that points in the same direction wrt wo
  if(normalMap)
  {
    Vec3f color = normalMap->evaluate(interaction);
    Vec3f texNormal = 2.0f * color - Vec3f::Ones();
    normalShading = normalize(interaction.tangent * texNormal[0] + interaction.bitangent * texNormal[1] + interaction.normalGeometric * texNormal[2]);
  }
  else
  {
    normalShading = interaction.normalShadingDefault;
  }
  float sideNsWo = signSide(dot(normalShading, wo));
  if(sideNgWo != sideNsWo)
  {
    normalShading = interaction.normalGeometric; // this is not consistent but good enough for the hack called normal map
  }
  // here we guarantuee signSide(normalShading, wo) == signSide(interaction.normalGeometric, wo)
  // but now we flip it according to normalGeometric:
  normalShading *= sideNgWo;
  return normalShading;
}

MaterialEvalResult PrincipledBRDF::evaluate(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag evalFlag)
{
  // There are two things to consider: leakage which we are trying to prevent with the statement above
  // and when wierd situations appear because geometric normal != shading normal
  //   there are two problematic subcases for the second case: wo is on the "inside" of the shading normal
  //          and the sampled out direction points into  the geometric normal
  //  see "Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing" for an overview
  // we ignore conservation loses and from the results in the paper and due to simplicity, address the first subcase using flipping , see sec. 3.4
  // the second subcase we just do a reflection across the normalGeometric surface

  float dotNgWo = dot(interaction.normalGeometric, wo);
  Vec3f normalShading = getShadingNormal(interaction, wo, dotNgWo);
  float dotNsWo = dot(normalShading, wo);
  float dotNsWi = dot(normalShading, wi);

  MaterialEvalResult result = MaterialEvalResult(Vec3f::Zero(), Vec3f::Zero(), normalShading);

  Vec3f albedoHit = albedo->evaluate(interaction);
  float roughnessHit = roughness->evaluate(interaction);
  float alpha = PrincipledBRDF::computeAlpha(roughnessHit);
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float anisotropyHit = this->anisotropy;
  float transmissionHit = transmission;

  float normalScale = computeNormalScale(wi, normalShading, interaction.normalGeometric);

  if(has(evalFlag, ShaderEvalFlag::DIFFUSE))
  {
    result.fDiffuse = PrincipledBRDF::evaluateDiffuse(dotNsWi, albedoHit, normalScale, metallicHit, specularHit, transmissionHit); // we use a Lambertian model here to keep it energy conserving
  }

  if(has(evalFlag, ShaderEvalFlag::CONCENTRATED))
  {
    Vec3f FDaccurate = Vec3f::Fill(std::numeric_limits<double>::signaling_NaN()); // not used since we can't hit dirac distribution (TODO change when adding roughness to glass/transmissionHit sub BRDF) and then we will need to compute multiple FD (one for each wavelength)

    float alpha_t, alpha_b;
    Vec3f tangent, bitangent;
    PrincipledBRDF::computeAnisotropyParameters(interaction, normalShading, alpha, anisotropyHit, alpha_t, alpha_b, tangent, bitangent);

    result.fConcentrated = PrincipledBRDF::evaluateConcentratedWithoutTransmission(
      dotNsWi,
      albedoHit, metallicHit, specularHit, roughnessHit, anisotropyHit, transmissionHit,
      wo, wi, alpha, alpha_t, alpha_b, tangent, bitangent, normalShading,
      normalScale, false, FDaccurate);
  }
  assertFinite(result.fDiffuse);
  assertFinite(result.fConcentrated);
  return result;
}

SampleResult PrincipledBRDF::sample(HASTY_INOUT(RNG) rng, const SurfaceInteraction& interaction, const Vec3f& wo, Wavelength wavelength, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag evalFlag)
{
  assert(wavelength != -1 || !this->varyIOR);

  Vec3f albedoHit = albedo->evaluate(interaction);
  float roughnessHit = roughness->evaluate(interaction);
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float anisotropyHit = this->anisotropy;
  float transmissionHit = transmission;
  Vec3f normalShading = getShadingNormal(interaction, wo, dot(interaction.normalGeometric, wo));

  float indexOfRefractionMat = this->getIndexOfRefraction(wavelength);

  return PrincipledBRDF::sample(
    rng, interaction, wo, normalShading,
    albedoHit,
    metallicHit, specularHit, roughnessHit, anisotropyHit, transmissionHit,
    wavelength, indexOfRefractionOutside, indexOfRefractionMat, adjoint, evalFlag);
}

float PrincipledBRDF::evaluateSamplePDF(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float outsideIOR)
{
  float dotNgWo = dot(interaction.normalGeometric, wo);
  Vec3f normalShading = getShadingNormal(interaction, wo, dotNgWo);
  float dotNsWo = dot(normalShading, wo);

  float roughnessHit = roughness->evaluate(interaction);
  float alpha = PrincipledBRDF::computeAlpha(roughnessHit);
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float transmissionHit = this->transmission;
  float anisotropyHit = this->anisotropy;

  float indexOfRefractionOutside = outsideIOR;
  float indexOfRefractionMat = this->indexOfRefraction;

  bool fromOutside = sideOutside(dotNgWo);
  float indexOfRefraction_o = fromOutside ? indexOfRefractionOutside : indexOfRefractionMat;
  float indexOfRefraction_t = fromOutside ? indexOfRefractionMat : indexOfRefractionOutside;

  PrincipledBRDF::ProbabilityResult probabilityResult = PrincipledBRDF::computeProbability(metallicHit, specularHit, transmissionHit, dotNsWo, indexOfRefraction_o, indexOfRefraction_t, ShaderEvalFlag::ALL);

  if(alpha == 0.0f)
  {
    float pdfDiffuse = evaluateHemisphereCosImportancePDF(normalShading, wi);
    float samplePd = probabilityResult.pDiffuseStrategy * pdfDiffuse;
    assertFinite(samplePd);
    return samplePd;
  }
  else
  {
    float alpha_t, alpha_b;
    Vec3f tangent, bitangent;
    PrincipledBRDF::computeAnisotropyParameters(interaction, normalShading, alpha, anisotropyHit, alpha_t, alpha_b, tangent, bitangent);
    float pdfSpec = sampleGGXVNDFGlobalDensity(normalShading, alpha_t, alpha_b, tangent, bitangent, wo, wi);
    float pdfDiffuse = evaluateHemisphereCosImportancePDF(normalShading, wi);
    float samplePd = probabilityResult.pSpecularStrategy * pdfSpec + probabilityResult.pDiffuseStrategy * pdfDiffuse;
    assertFinite(samplePd);
    return samplePd;
  }
}
bool PrincipledBRDF::hasDiffuseLobe(const SurfaceInteraction& interaction)
{
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float transmissionHit = transmission;

  return (1.0f - 0.5f * specularHit) * (1.0 - transmissionHit) * (1.0 - metallicHit) > 0.0f;
}

}
