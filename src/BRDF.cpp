#include <Hasty/BRDF.h>

#include <Hasty/Scene.h>
#include <Hasty/PathTracing.h>

namespace Hasty
{

bool computeRefractionDirection(const Vec3f& wi, const Vec3f& normal, float indexOfRefraction_i, float indexOfRefraction_t, Vec3f* wt)
{
  // t stands for transmittance
  //https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#sec:specular-transmit
  float relIOR = indexOfRefraction_i / indexOfRefraction_t;
  float cos_i = dot(wi, normal);
  assert(cos_i >= 0.0f);
  //float side = cos_i >= 0.0f ? 1.0f : -1.0f;
  float sin_iSq = std::max(0.f, 1.f - cos_i * cos_i);
  float sin_tSq = (relIOR * relIOR) * sin_iSq;
  if (sin_tSq >= 1.0f)
  {
    // in this case we get total internal reflection and we can't compute cos_t (going complex)
    return false;
  }
  float cos_t = std::sqrt(1.0f - sin_tSq);
  (*wt) = (-relIOR) * wi + (relIOR * cos_i - cos_t) * normal;
  wt->normalize();
  return true;
}

//https://psgraphics.blogspot.com/2016/12/bug-in-my-schlick-code.html?m=0
inline float fresnelSchlick(float cos_i, float indexOfRefraction_i, float indexOfRefraction_t)
{
  //https://en.wikipedia.org/wiki/Schlick%27s_approximation
  float R0 = square((indexOfRefraction_i - indexOfRefraction_t) / (indexOfRefraction_i + indexOfRefraction_t));
  float cosT = cos_i;
  return R0 + (1.0f - R0) * powci<5>(1.0f - cosT);
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
Vec3f sampleDGGX(RNG& rng, const Vec3f& normal, float alpha, const Vec3f& dir1, float* pDensity)
{
  if (dot(dir1, normal) < 0.0f)
  {
    std::cout << " warning sampleDGGX from the inside" << '\n';
    (*pDensity) = 1.0f;
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
    Vec3f rv(sintheta * cosphi, sintheta * sinphi, costheta);

    RotationBetweenTwoVectors<float, Vec3f> rotation(Vec3f(0.0f, 0.0f, 1.0f), normal);
    h = rotation * rv;
    h = normalize(h);
  } while (dot(h, normal) <= 0.0f);

  // p(dir2) = p( h  | dir1) / abs(det(jac_h(transformation)))
  // dir2 =  2 * h.dot(dir1) * h - dir1
  //  we can just use http://www.matrixcalculus.org/ for the jacobian, sth like: 2 *h * (h'* y) - y, giving  2 h y' + 2 h' y * Identity, determinant is 4 h' y, i guess
  Vec3f dir2 = reflectAcross(dir1, h);
  dir2 = normalize(dir2);

  float nh = dot(normal, h);
  float q = (nh * nh) * (alpha * alpha - 1.0f) + 1.0f;
  float pDensityH = alpha * alpha * nh / (float(Pi) * (q * q));
  (*pDensity) = pDensityH / std::abs(4.0f * dot(dir1, h));
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
  float lambda = 0.5f * ( std::sqrt(1.0f + (square(alpha_t * tv) +square(alpha_b * bv))/square(nv))  - 1.0f);
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
  if (normSq(T1) < 1e-12f) T1 = X;
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
Vec3f sampleGGXVNDFGlobal(RNG& rng, const Vec3f& normal, float alpha_t, float alpha_b, const Vec3f& tangent, const Vec3f& bitangent, const Vec3f& dir1, float* pDensity)
{
  if (dot(dir1, normal) <= 0.0f)
  {
    std::cout << " warning sampleGGXVNDFGlobal from the inside" << '\n';
    (*pDensity) = 0.0f;
    return -normal;
  }

  Vec3f NeGlobal;
  do
  {
    Vec3f Ve = Vec3f(dot(tangent, dir1), dot(bitangent, dir1), dot(normal, dir1));
    Vec3f Ne = sampleGGXVNDF(Ve, alpha_t, alpha_b, uniform01f(rng), uniform01f(rng));

    NeGlobal = tangent * Ne[0] + bitangent * Ne[1] + normal * Ne[2];
    NeGlobal = normalize(NeGlobal);
  } while (dot(NeGlobal, dir1) <= 0.0f);

  Vec3f dir2 = reflectAcross(dir1, NeGlobal);
  dir2 = normalize(dir2);
  // V = dir1, N = H
  float G1 = smithGGX(dot(normal, dir1), dot(tangent, dir1), dot(bitangent, dir1), alpha_t, alpha_b);
  float D_ggx = DGGX(dot(normal, NeGlobal), dot(tangent, NeGlobal), dot(bitangent, NeGlobal), alpha_t, alpha_b);
  float pDensityH = G1 * dot(dir1, NeGlobal) * D_ggx / dot(dir1, normal);
  (*pDensity) = pDensityH / (1e-7f + 4.0f * std::abs(dot(dir1, NeGlobal)));
  return dir2;
}
float sampleGGXVNDFGlobalDensity(const Vec3f& normal, float alpha_t, float alpha_b, const Vec3f& tangent, const Vec3f& bitangent, const Vec3f& dir1, const Vec3f& dir2)
{
  if (dot(dir1, normal) <= 0.0f)
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
inline float computeNormalScale(const SurfaceInteraction& interaction, const Vec3f& wi, const Vec3f& normalShading) {
  return std::max(0.0f, dot(wi, normalShading)) / (1e-7f + std::abs(dot(wi, interaction.normalGeometric)));
}

MaterialEvalResult GlassBTDF::evaluate(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag evalFlag)
{
  MaterialEvalResult result(Vec3f::Zero(), Vec3f::Zero(), interaction.normalShadingDefault);

  if (has(evalFlag, ShaderEvalFlag::DIFFUSE))
  {
    // Zero
  }
  if (has(evalFlag, ShaderEvalFlag::SPECULAR))
  {
    MaterialEvalResult mspec = evaluateSpecular(interaction, wo, wi, interaction.normalGeometric, result.normalShading, indexOfRefractionOutside, adjoint);
    result.fSpecular += mspec.fSpecular;
  }

  return result;
}
float GlassBTDF::evaluateSpecular(const Vec3f& wo, const Vec3f& wi, const Vec3f &normalGeometric, const Vec3f& normal, float indexOfRefractionOutside, float indexOfRefractionMat, bool adjoint)
{
  auto evaluateDir = [&normal, &wi, &normalGeometric](const Vec3f& dir1, const Vec3f& dir2) {
    bool outside = dir1.dot(normalGeometric) > 0.0f;
    bool outsideShadingNormal = dir1.dot(normal) > 0.0f;
    if (outside != outsideShadingNormal)
    {
      return false;
    }

    return (dir1 - dir2).normL1() < 1e-5;
  };
  float indexOfRefraction_i, indexOfRefraction_t;
  float cos_i = normal.dot(wi);// the formulas on the internet seem to rely on  the fact that the normal is aligned with wi so we introduce a new variable normal_i
  
  bool outside = wo.dot(normalGeometric) > 0.0f;
  bool outsideShadingNormal = wo.dot(normal) > 0.0f;
  if (outside != outsideShadingNormal
    || (wi.dot(normalGeometric) > 0.0f) != (wi.dot(normal) > 0.0f))
  {
    return 0.0f;
  }

  Vec3f normal_i;
  if (cos_i > 0.0f)  // here i is where light is coming from, and t is the opposite (note that t can be unequal to o)
  {
    indexOfRefraction_i = indexOfRefractionOutside;
    indexOfRefraction_t = indexOfRefractionMat;
    normal_i = normal;
  }
  else
  {
    indexOfRefraction_i = indexOfRefractionMat;
    indexOfRefraction_t = indexOfRefractionOutside;
    normal_i = -normal;
    cos_i = -cos_i;
  }

  Vec3f refracDir;
  bool refraction = computeRefractionDirection(wi, normal_i, indexOfRefraction_i, indexOfRefraction_t, &refracDir);
  // we have the following phenomena; total internal reflection, partial reflection just in general which is given by fresnel term, and refraction for the rest

  float f = 0.0f;

  if (!refraction)
  {
    //total internal reflection,
    Vec3f reflecDir = reflectAcross(wi, normal_i);
    f += evaluateDir(reflecDir, wo) ? 1.0f : 0.0f;
  }
  else
  {
    // refraction, now we consider fresnel split
    float cos_t = std::abs(refracDir.dot(normal_i));
    float F = fresnelDielectric(cos_i, cos_t, indexOfRefraction_i, indexOfRefraction_t); // goes from 0 to 1 for cos_i going from 0 to 90 degrees, at higher degrees we get more reflection

    // try reflection
    Vec3f reflecDir = reflectAcross(wi, normal_i);
    f += evaluateDir(reflecDir, wo) ? F : 0.0f;
    float scale = adjoint ? 1.0f : square(indexOfRefraction_t) / square(indexOfRefraction_i);
    f += evaluateDir(refracDir, wo) ? (1.0f - F) * scale : 0.0f; // See "Non-symmetric Scattering in Light Transport Algorithms" the eq. after eq. 10
  }

  return f;
}
MaterialEvalResult GlassBTDF::evaluateSpecular(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, const Vec3f &normalGeometric, const Vec3f& normal, float indexOfRefractionOutside, bool adjoint)
{
  Vec3f f;
  if (varyIOR)
  {
    f = Vec3f(
      evaluateSpecular(wo, wi, normalGeometric, normal, indexOfRefractionOutside, getIndexOfRefraction(0), adjoint),
      evaluateSpecular(wo, wi, normalGeometric, normal, indexOfRefractionOutside, getIndexOfRefraction(1), adjoint),
      evaluateSpecular(wo, wi, normalGeometric, normal, indexOfRefractionOutside, getIndexOfRefraction(2), adjoint));
  }
  else
  {
    float f0 = evaluateSpecular(wo, wi, normalGeometric, normal, indexOfRefractionOutside, getIndexOfRefraction(0), adjoint);
    f = Vec3f(f0, f0, f0);
  }
  return MaterialEvalResult(Vec3f::Zero(), f, normal);
}

SampleResult GlassBTDF::sample(RNG& rng, const SurfaceInteraction& interaction, const LightRayInfo& lightRay, const Vec3f& wOut, OutsideIORFunctor getOutsideIOR, bool adjoint, ShaderEvalFlag evalFlag)
{
  // We interpet evalFlag :: ALL = whole distribution
  assert(evalFlag != ShaderEvalFlag::NONE); // no distribution to sample
  assert(evalFlag != ShaderEvalFlag::DIFFUSE); // no distribution to sample
  SampleResult result;
  result.lightRay = lightRay;
  result.direction = this->sampleSpecular(rng, interaction, result.lightRay, wOut, getOutsideIOR, adjoint, &result.throughputDiffuse, &result.throughputSpecular, &result.pdfOmega, &result.outside);
  return result;
}
Vec3f GlassBTDF::sampleSpecular(RNG& rng, const SurfaceInteraction& interaction, LightRayInfo& lightRay, const Vec3f& wo, OutsideIORFunctor getOutsideIOR, bool adjoint, Vec3f* throughputDiffuse, Vec3f* throughputSpecular, float* pDensity, bool *outside)
{
  float indexOfRefractionMat;
  if (varyIOR)
  {
    int wavelength = lightRay.getWavelength(rng, albedo.cwiseAbs());
    indexOfRefractionMat = getIndexOfRefraction(wavelength);
  }
  else
  {
    indexOfRefractionMat = getIndexOfRefraction(0);
  }
  float indexOfRefractionOutside = getOutsideIOR();

  auto sampleDirection = [](RNG& rng, const Vec3f& dir, float* p) {
    (*p) = 1e9f;
    return dir;
  };

  Vec3f normal = interaction.normalShadingDefault;

  float indexOfRefraction_i, indexOfRefraction_t;
  float cos_o = normal.dot(wo);// the formulas on the internet seem to rely on  the fact that the normal is aligned with wi so we introduce a new variable normal_i
  (*outside) = wo.dot(interaction.normalGeometric) > 0.0f;
  bool outsideShadingNormal = wo.dot(normal) > 0.0f;
  if ((*outside) != outsideShadingNormal)
  {
    (*throughputDiffuse) = Vec3f::Zero();
    (*throughputSpecular) = Vec3f::Zero();
    (*pDensity) = 1.0f;
    return normal;
  }

  Vec3f normal_o;
  if (*outside)  // here i is where light is coming from, and t is the opposite (note that t can be unequal to o)
  {
    indexOfRefraction_i = indexOfRefractionOutside;
    indexOfRefraction_t = indexOfRefractionMat;
    normal_o = normal;
  }
  else
  {
    indexOfRefraction_i = indexOfRefractionMat;
    indexOfRefraction_t = indexOfRefractionOutside;
    normal_o = -normal;
    cos_o = -cos_o;
  }

  Vec3f refracDir;
  bool refraction = computeRefractionDirection(wo, normal_o, indexOfRefraction_i, indexOfRefraction_t, &refracDir);
  if (!refraction)
  {
    Vec3f reflecDir = reflectAcross(wo, normal_o);
    assertUnitLength(reflecDir);
    Vec3f sampleDir = sampleDirection(rng, reflecDir, pDensity);
    MaterialEvalResult evalResult = evaluateSpecular(interaction, wo, sampleDir, interaction.normalGeometric, normal, indexOfRefractionOutside, adjoint);
    (*throughputDiffuse) = Vec3f::Zero();
    (*throughputSpecular) = evalResult.fSpecular;
    return sampleDir;
  }
  else
  {
    float cos_t = std::abs(refracDir.dot(normal_o));
    float F = fresnelDielectric(cos_o, cos_t, indexOfRefraction_i, indexOfRefraction_t); // goes from 0 to 1 for cos_i going from 0 to 90 degrees, at higher degrees we get more reflection

    // try reflection
    if (uniform01f(rng) < F)
    {
      Vec3f reflecDir = reflectAcross(wo, normal_o);
      assertUnitLength(reflecDir);
      Vec3f sampleDir = sampleDirection(rng, reflecDir, pDensity);
      (*pDensity) *= F;
      MaterialEvalResult evalResult = evaluateSpecular(interaction, wo, sampleDir, interaction.normalGeometric, normal, indexOfRefractionOutside, adjoint);
      (*throughputDiffuse) = Vec3f::Zero();
      (*throughputSpecular) = evalResult.fSpecular / F;
      return sampleDir;
    }
    else
    {
      // refraction: switch outside
      (*outside) = !(*outside);
      assertUnitLength(refracDir);
      Vec3f sampleDir = sampleDirection(rng, refracDir, pDensity);
      (*pDensity) *= (1.0f - F);
      MaterialEvalResult evalResult = evaluateSpecular(interaction, wo, sampleDir, interaction.normalGeometric, normal, indexOfRefractionOutside, adjoint);
      (*throughputDiffuse) = Vec3f::Zero();
      (*throughputSpecular) = evalResult.fSpecular / (1.0f - F);
      return sampleDir;
    }
  }
}

float GlassBTDF::evaluateSamplePDF(const SurfaceInteraction &interaction, const Vec3f& wo, const Vec3f& wi)
{
  return 0.0f;
}
bool GlassBTDF::hasDiffuseLobe(const SurfaceInteraction& interaction)
{
  return false;
}
float GlassBTDF::getIndexOfRefraction(int wavelength) const
{
  if (!varyIOR) return indexOfRefraction;
  switch (wavelength)
  {
  case 0: return indexOfRefraction * (1.43f / 1.45f);
  case 1: return indexOfRefraction * (1.45f / 1.45f);
  case 2: return indexOfRefraction * (1.47f / 1.45f);
  default: return std::numeric_limits<float>::signaling_NaN();
  }
}


PrincipledBRDF::PrincipledBRDF(std::unique_ptr<ITextureMap3f> albedo, std::unique_ptr<ITextureMap1f> roughness, std::unique_ptr<ITextureMap1f> metallic, float specular, float indexOfRefraction, std::unique_ptr<ITextureMap3f> normalMap, float anisotropy)
  :albedo(std::move(albedo)), roughness(std::move(roughness)), metallic(std::move(metallic)), specular(specular), IOR(indexOfRefraction), normalMap(std::move(normalMap)), anisotropy(anisotropy)
{
}
PrincipledBRDF::~PrincipledBRDF()
{

}
Vec3f PrincipledBRDF::getShadingNormal(const SurfaceInteraction& interaction, const Vec3f& wo)
{
  Vec3f normalShading;
  if (normalMap)
  {
    Vec3f color = normalMap->evaluate(interaction);
    Vec3f texNormal = 2.0f * color - Vec3f::Ones();
    normalShading = normalize(interaction.tangent * texNormal[0] + interaction.bitangent * texNormal[1] + interaction.normalGeometric * texNormal[2] );
    if (dot(wo, normalShading) <= 0.0f)
    {
      normalShading = interaction.normalGeometric; // this is not consistent but good enough for the hack called normal map
    }
  }
  else
  {
    normalShading = interaction.normalShadingDefault;
  }
  return normalShading;
}
float PrincipledBRDF::computeAlpha(float roughness)
{
  //we have alpha min to not have dirac distribution as special case
  float alpha = roughness * roughness;
  if (alpha < 1e-2f && alpha != 0.0f) // we handle alpha == 0.0f specifically
  {
    //std::cout << " warning: roughness very small, clamping due to instabilities " << std::endl;
    alpha = 1e-2f;
  }
  return alpha;
}
void PrincipledBRDF::computeProbability(float metallicHit, float specularHit, float* diffSpecMix, float* pProbablitySpec)
{
  *diffSpecMix = std::min(1.0f - metallicHit, 1.0f - 0.5f * specularHit); // we want this to be zero when metallic and one when specular = 0
  *pProbablitySpec = 1.0f - *diffSpecMix;
}
void PrincipledBRDF::computeAnisotropyParameters(const SurfaceInteraction &interaction, const Vec3f& normalShading, float alpha, float &alpha_t, float &alpha_b, Vec3f &tangent, Vec3f &bitangent)
{
  float aspect = std::sqrt(1.0f - 0.9f * anisotropy);
  alpha_t = alpha / aspect;
  alpha_b = alpha * aspect;
  RotationBetweenTwoVectors<float, Vec3f> rotation(interaction.normalGeometric, normalShading);
  tangent = rotation * interaction.tangent;
  bitangent = rotation * interaction.bitangent;
}

Vec3f PrincipledBRDFevaluateDiffuse(const Vec3f &albedoHit, float normalScale, float diffuseSpecMix) {
  return albedoHit * float(InvPi) * normalScale * diffuseSpecMix;
}
Vec3f PrincipledBRDFevaluateSpecular(
  const Vec3f& albedoHit, float metallicHit, float specularHit,
  float roughnessHit, float anisotropyHit,
  const Vec3f& wo, const Vec3f& wi,
  float alpha,
  float alpha_t, float alpha_b,
  const Vec3f& tangent, const Vec3f& bitangent,
  const Vec3f& normalShading, float normalScale,
  float diffuseSpecMix) {
  Vec3f h = normalize(wo + wi);
  float lh = std::abs(dot(wi, h));
  //Pseudo fresnel:
  // https://www.youtube.com/watch?v=kEcDbl7eS0w basically F0 is directly the color one desires in the middle, and it's actually more principled to use F0 + SChlick than using the full fresnel equations because we don't do proper spectral rendering
  Vec3f Ks = Vec3f::Ones();
  Vec3f F0 = mix(Ks, albedoHit, metallicHit); // and we also have a specular part for non metalics, in that case it's kinda white, wheras for a metallic object the light gets "colored"

  Vec3f FH = F0 + (Vec3f::Ones() - F0) * powci<5>(1.0f - lh); // here if we are at a grazing angle, we  remove some F0, and add some "achromatic" reflectance (Vec3f::ones)

  float nh = dot(normalShading, h);

  float nv = std::abs(dot(normalShading, wo));
  float nl = std::abs(dot(normalShading, wi));
  float Go = smithGGX(nv, dot(tangent, wo), dot(bitangent, wo), alpha_t, alpha_b);
  float Gi = smithGGX(nl, dot(tangent, wi), dot(bitangent, wi), alpha_t, alpha_b);
  float G = Go * Gi;

  float D_ggx;

  if (alpha == 0.0f)
  {
    // D_GGX explodes for alpha= 0 and nh = 1
    // we can find this by using importance sample formula and then let alpha go to zero?, G is always 1 in that case
    // => \int F D G/ (4 nv nl) dl ~= 1/N \sum_i  F D Gi Go/ (4 nv nl_i  p(l_i) )
    //           ~= 1/N \sum_i  F / nv ~=   F / nv for alpha = 0
    D_ggx = std::abs(std::abs(nh) - 1.0f) < 1e-3f ? 4.0f * nl : 0.0f;
  }
  else
  {
    D_ggx = DGGX(nh, dot(tangent, h), dot(bitangent, h), alpha_t, alpha_b);
  }

  Vec3f result = FH * (D_ggx * G * normalScale / std::abs(1e-7f + 4.0f * nv * nl) * (1.0f - diffuseSpecMix));
  return result;
}
MaterialEvalResult PrincipledBRDF::evaluate(const SurfaceInteraction& interaction, const Vec3f& wo, const Vec3f& wi, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag evalFlag)
{
  // There are two things to consider: leakage which we are trying to prevent with the statement above
  // and when wierd situations appear because geometric normal != shading normal
  //   there are two problematic subcases for the second case: wo is on the "inside" of the shading normal
  //          and the sampled out direction points into  the geoemtric normal
  //  see "Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing" for an overview
  // we ignore conservation loses and from the results in the paper and due to simplicity, address the first subcase using flipping , see sec. 3.4
  // the second subcase we just do a reflection across the normalGeometric surface
  
  Vec3f normalShading = getShadingNormal(interaction, wo);
  // no leakage:
  // the following destroys the results if we add an offset to the ray origin to try to solve the shadow terminator problem...
  //if ( wi.dot(normalGeometric) <= 0.0f) return MaterialEvalResult(Vec3f::Zero(), normalShading);
  if (dot(wo, normalShading) <= 0.0f)
  {
    return MaterialEvalResult(Vec3f::Zero(), Vec3f::Zero(), normalShading);
  }
  MaterialEvalResult result = MaterialEvalResult(Vec3f::Zero(), Vec3f::Zero(), normalShading);

  Vec3f albedoHit = albedo->evaluate(interaction);
  float roughnessHit = roughness->evaluate(interaction);
  float alpha = computeAlpha(roughnessHit);
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float anisotropyHit = this->anisotropy;
  float diffuseSpecMix, pSpecularStrategy;
  computeProbability(metallicHit, specularHit, &diffuseSpecMix, &pSpecularStrategy);

  float normalScale = std::max(0.0f, dot(wi, normalShading)) / (1e-7f + std::abs(dot(wi, interaction.normalGeometric)));

  if (has(evalFlag, ShaderEvalFlag::DIFFUSE))
  {
    result.fDiffuse = PrincipledBRDFevaluateDiffuse(albedoHit, normalScale, diffuseSpecMix); // we use a Lambertian model here to keep it energy conserving
  }

  if (has(evalFlag, ShaderEvalFlag::SPECULAR))
  {
    float alpha_t, alpha_b;
    Vec3f tangent, bitangent;
    computeAnisotropyParameters(interaction, normalShading, alpha, alpha_t, alpha_b, tangent, bitangent);

    result.fSpecular = PrincipledBRDFevaluateSpecular(albedoHit, metallicHit, specularHit, roughnessHit, anisotropyHit, wo, wi, alpha, alpha_t, alpha_b, tangent, bitangent, normalShading, normalScale, diffuseSpecMix);
  }
  assertFinite(result.fDiffuse);
  assertFinite(result.fSpecular);
  return result;
}

SampleResult PrincipledBRDF::sample(RNG& rng, const SurfaceInteraction& interaction, const LightRayInfo& lightRay, const Vec3f& wOut, OutsideIORFunctor getOutsideIOR, bool adjoint, ShaderEvalFlag evalFlag)
{
  SampleResult result;

  assert(evalFlag != ShaderEvalFlag::NONE); // no distribution to sample
  assert(evalFlag != ShaderEvalFlag::DIFFUSE); // not implemented
  if (evalFlag == ShaderEvalFlag::SPECULAR)
  {
    result.direction = this->sampleSpecular(rng, interaction, lightRay, wOut, getOutsideIOR, adjoint, &result.throughputDiffuse, &result.throughputSpecular, &result.pdfOmega, &result.outside);
    return result;
  }
  result.outside = dot(wOut, interaction.normalGeometric) > 0.0f;
  if (!result.outside)
  {
    result.pdfOmega = 1.0;
    result.throughputDiffuse = Vec3f::Zero();
    result.throughputSpecular = Vec3f::Zero();
    result.direction = -interaction.normalGeometric; // returning any direction, cause there should be no value that is nonzero
    return result;
  }

  float outsideIOR = getOutsideIOR();

  Vec3f normalShading = getShadingNormal(interaction, wOut);
  if (dot(wOut, normalShading) <= 0.0f)
  {
    result.pdfOmega = 1.0;
    result.throughputDiffuse = Vec3f::Zero();
    result.throughputSpecular = Vec3f::Zero();
    result.direction = -interaction.normalGeometric; // returning any direction, cause there should be no value that is nonzero
    return result;
  }

  Vec3f albedoHit = albedo->evaluate(interaction);
  float roughnessHit = roughness->evaluate(interaction);
  float alpha = computeAlpha(roughnessHit);
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float anisotropyHit = this->anisotropy;
  float diffuseSpecMix, pSpecularStrategy;
  computeProbability(metallicHit, specularHit, &diffuseSpecMix, &pSpecularStrategy);

  float alpha_t, alpha_b;
  Vec3f tangent, bitangent;
  computeAnisotropyParameters(interaction, normalShading, alpha, alpha_t, alpha_b, tangent, bitangent);

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

  bool useSpecularStrategy = uniform01f(rng) < pSpecularStrategy;
  if (useSpecularStrategy)
  {

    if (alpha == 0.0f) // delta distribution; need to keep wierd fake measure
    {
      //F' = {
      //      (f_specular L cos(omegaSpec)) / pSpecStr, omega = omegaSpec     xi < pSpecStr
      //      f_diffuse L cos(omega) / ((1 - pSpecStr) p_diff), omega ~ p_diff xi >= pSpecStr
      //     }
      //E[F'] = pSpecStr f_specular(omega) L(omega) cos(omegaSpec) / pSpecStr + (1 - pSpecStr) \int f_diffuse L cos(omega) / ((1 - pSpecStr) pDiffuse) pDiffuse(omega) domega
      //       = L f_specular(omega_specular) cos(omega_specular) + \int L cos(omega) f_diffuse(omega) domega
      //       = \int  L cos(omega) f_specular dirac_specular(omega) domega + \int L cos(omega) f_diffuse domega
      //       = \int  L cos(omega) (f_specular dirac_specular(omega)  +  f_diffuse) domega
      result.direction = reflectAcross(wOut, normalShading);
      result.pdfOmega = pSpecularStrategy;
      const Vec3f& wi = result.direction;
      float normalScale = computeNormalScale(interaction, wi, normalShading);
      evalResult.fDiffuse = Vec3f::Zero();
      evalResult.fSpecular = PrincipledBRDFevaluateSpecular(
        albedoHit, metallicHit, specularHit,
        roughnessHit, anisotropyHit,
        wOut, result.direction,
        alpha,
        alpha_t, alpha_b,
        tangent, bitangent,
        normalShading, normalScale, diffuseSpecMix);
    }
    else
    {
      float pdfSpec;
      result.direction = sampleGGXVNDFGlobal(rng, normalShading, alpha_t, alpha_b, tangent, bitangent, wOut, &pdfSpec);
      float pdfDiffuse = evaluateHemisphereCosImportancePDF(normalShading, result.direction);

      result.pdfOmega = pSpecularStrategy * pdfSpec + (1.0f - pSpecularStrategy) * pdfDiffuse;
      const Vec3f& wi = result.direction;
      float normalScale = computeNormalScale(interaction, wi, normalShading);
      evalResult.fDiffuse = PrincipledBRDFevaluateDiffuse(albedoHit, normalScale, diffuseSpecMix);
      evalResult.fSpecular = PrincipledBRDFevaluateSpecular(
        albedoHit, metallicHit, specularHit,
        roughnessHit, anisotropyHit,
        wOut, result.direction,
        alpha,
        alpha_t, alpha_b,
        tangent, bitangent,
        normalShading, normalScale, diffuseSpecMix);
    }
  }
  else
  {
    float pdfDiffuse;
    result.direction = sampleHemisphereCosImportance(rng, normalShading, &pdfDiffuse);
    const Vec3f& wi = result.direction;
    float normalScale = computeNormalScale(interaction, wi, normalShading);

    if (alpha == 0.0f) // delta distribution; need to keep wierd fake measure
    {
      result.pdfOmega = (1.0f - pSpecularStrategy) * pdfDiffuse;
      evalResult.fDiffuse = PrincipledBRDFevaluateDiffuse(albedoHit, normalScale, diffuseSpecMix);
      evalResult.fSpecular = Vec3f::Zero();
    }
    else
    {
      float pdfSpec = sampleGGXVNDFGlobalDensity(normalShading, alpha_t, alpha_b, tangent, bitangent, wOut, result.direction);
      result.pdfOmega = pSpecularStrategy * pdfSpec + (1.0f - pSpecularStrategy) * pdfDiffuse;
      evalResult.fDiffuse = PrincipledBRDFevaluateDiffuse(albedoHit, normalScale, diffuseSpecMix);
      evalResult.fSpecular = PrincipledBRDFevaluateSpecular(
        albedoHit, metallicHit, specularHit,
        roughnessHit, anisotropyHit,
        wOut, result.direction,
        alpha,
        alpha_t, alpha_b,
        tangent, bitangent,
        normalShading, normalScale, diffuseSpecMix);
    }
  }
  float m = std::abs(dot(result.direction, interaction.normalGeometric)) / result.pdfOmega;
  result.throughputDiffuse = evalResult.fDiffuse * m;
  result.throughputSpecular = evalResult.fSpecular * m;

  assertUnitLength(result.direction);
  assertFinite(result.throughputDiffuse);
  assertFinite(result.throughputSpecular);
  assertFinite(result.pdfOmega);
  return result;
}
Vec3f PrincipledBRDF::sampleSpecular(RNG& rng, const SurfaceInteraction& interaction, const LightRayInfo& lightRay, const Vec3f& wOut, OutsideIORFunctor getOutsideIOR, bool adjoint, Vec3f* throughputDiffuse, Vec3f* throughputSpecular, float* pDensity, bool *outside)
{
  *outside = dot(wOut, interaction.normalGeometric) > 0.0f;
  if (!(*outside))
  {
    (*pDensity) = 1.0f;
    (*throughputDiffuse) = Vec3f::Zero();
    (*throughputSpecular) = Vec3f::Zero();
    return -interaction.normalGeometric; // returning any direction, cause there should be no value that is nonzero
  }
  
  float roughnessHit = roughness->evaluate(interaction);
  float alpha = computeAlpha(roughnessHit);

  Vec3f normalShading = getShadingNormal(interaction, wOut);
  if (dot(wOut, normalShading) < 0.0f)
  {
    (*pDensity) = 1.0f;
    (*throughputDiffuse) = Vec3f::Zero();
    (*throughputSpecular) = Vec3f::Zero();
    return -interaction.normalGeometric; // returning any direction, cause there should be no value that is nonzero
  }
  float outsideIOR = getOutsideIOR();

  if (alpha == 0.0f) // dirac specular distribution
  {
    Vec3f direction2 = reflectAcross(wOut, normalShading);
    MaterialEvalResult evalResult = this->evaluate(interaction, wOut, direction2, outsideIOR, adjoint, ShaderEvalFlag::SPECULAR);
    float m = std::abs(dot(direction2, interaction.normalGeometric));
    (*throughputDiffuse) = evalResult.fDiffuse * m;
    (*throughputSpecular) = evalResult.fSpecular * m;
    (*pDensity) = 1e9f;
    
    assertFinite(*throughputDiffuse);
    assertFinite(*throughputSpecular);
    assertFinite(*pDensity);
    return direction2;
  }
  else
  {
    float alpha_t, alpha_b;
    Vec3f tangent, bitangent;
    computeAnisotropyParameters(interaction, normalShading, alpha, alpha_t, alpha_b, tangent, bitangent);
    Vec3f direction2 = sampleGGXVNDFGlobal(rng, normalShading, alpha_t, alpha_b, tangent, bitangent, wOut, pDensity);
    MaterialEvalResult evalResult = this->evaluate(interaction, wOut, direction2, outsideIOR, adjoint, ShaderEvalFlag::SPECULAR);
    float mp = std::abs(dot(direction2, interaction.normalGeometric)) / (*pDensity);
    (*throughputDiffuse) = evalResult.fDiffuse * mp;
    (*throughputSpecular) = evalResult.fSpecular * mp;

    assertFinite(*throughputDiffuse);
    assertFinite(*throughputSpecular);
    assertFinite(*pDensity);
    return direction2;
  }
}

float PrincipledBRDF::evaluateSamplePDF(const SurfaceInteraction &interaction, const Vec3f& wo, const Vec3f& wi)
{
  Vec3f normalShading = getShadingNormal(interaction, wo);
  if (dot(wo, interaction.normalGeometric) <= 0.0f || dot(wo, normalShading) <= 0.0f)
  {
    return 0.0f;
  }
  
  float roughnessHit = roughness->evaluate(interaction);
  float alpha = computeAlpha(roughnessHit);
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float diffuseSpecMix, pSpecularStrategy;
  computeProbability(metallicHit, specularHit, &diffuseSpecMix, &pSpecularStrategy);

  if (alpha == 0.0f) // delta distribution; need to keep wierd fake measure
  {
    //assuming that we never call this with the specular direction....
    float pdfDiffuse = evaluateHemisphereCosImportancePDF(normalShading, wi);
    float samplePd = (1.0f - pSpecularStrategy) * pdfDiffuse;
    assertFinite(samplePd);
    return samplePd;
  }
  else
  {
    float alpha_t, alpha_b;
    Vec3f tangent, bitangent;
    computeAnisotropyParameters(interaction, normalShading, alpha, alpha_t, alpha_b, tangent, bitangent);
    float pdfSpec = sampleGGXVNDFGlobalDensity(normalShading, alpha_t, alpha_b, tangent, bitangent, wo, wi);
    float pdfDiffuse = evaluateHemisphereCosImportancePDF(normalShading, wi);
    float samplePd = pSpecularStrategy * pdfSpec + (1.0f - pSpecularStrategy) * pdfDiffuse;
    assertFinite(samplePd);
    return samplePd;
  }
}
bool PrincipledBRDF::hasDiffuseLobe(const SurfaceInteraction& interaction)
{
  float metallicHit = metallic->evaluate(interaction);
  float specularHit = this->specular;
  float diffuseSpecMix, pSpecularStrategy;
  computeProbability(metallicHit, specularHit, &diffuseSpecMix, &pSpecularStrategy);

  return pSpecularStrategy < 1.0f;
}

}
