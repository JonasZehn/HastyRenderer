#include <Hasty/PhotonMapping.h>

#include <Hasty/Json.h>
#include <Hasty/MIS.h>

#include <nanoflann.hpp>

namespace Hasty
{

class Photon
{
public:
  Vec3f wi;
  float ngwi;
  Vec3f x;
  Vec3f flux;
  Photon(const Ray& ray, float ngwi, const Vec3f& x, const Vec3f& flux)
    : wi(-ray.direction()), ngwi(ngwi), x(x), flux(flux)
  {

  }
};
class PhotonSearchResult
{
public:
  friend class PhotonMap;

  PhotonSearchResult()
  {

  }

  std::size_t size() const
  {
    return idcs.size();
  }

  const Photon& operator[](std::size_t i)
  {
    return photons->operator[](idcs[i]);
  }
private:
  std::vector<uint32_t> idcs;
  const std::vector<Photon>* photons;
};

struct PointCloudf
{
  using Point = Vec3f;

  using coord_t = float;

  const std::vector<Point>& pts;
  PointCloudf(const std::vector<Point>& p) : pts(p) {}

  inline size_t kdtree_get_point_count() const { return pts.size(); }

  inline float kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    return pts[idx][dim];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const
  {
    return false;
  }
};

class PhotonMap
{
public:

  PhotonMap()
    :tree(my_kd_tree_t(3, cloud))
  {
    m_emittedCount = 0;
  }

  void clear(bool useHashCells, float cellSize)
  {
    m_emittedCount = 0;
    photons.clear();
    points.clear();

    m_useHashCells = useHashCells;

    if(m_useHashCells) hashCells.clear(cellSize);
  }
  void addPhoton(const Ray& ray, const RayHit& hit, const Vec3f& flux)
  {
    photons.emplace_back(ray, hit.interaction.normalGeometric.dot(hit.wFrom()), hit.interaction.x, flux);
    points.push_back(hit.interaction.x);
  }

  void build()
  {
    if (m_useHashCells)
    {
      hashCells.initialize(points);
    }
    else
    {
      tree.buildIndex();
    }
  }

  void printStats()
  {
    if (m_useHashCells)
    {
      hashCells.printStats();
    }
  }

  void radiusSearch(const Vec3f& center, float radius, PhotonSearchResult* result)
  {
    if (m_useHashCells)
    {
      result->photons = &photons;
      hashCells.radiusNeighbors(center, radius, result->idcs);
    }
    else
    {
      result->photons = &photons;
    
      nanoflann::SearchParams params;
      params.sorted = false;
      const size_t nMatches = tree.radiusSearch(&center[0], adaptorConvertRadius(radius), ret_matches, params);
      result->idcs.clear();
      for (const auto & p : ret_matches)
      {
        uint32_t index = p.first;
        result->idcs.push_back(index);
      }
    }
  }
  template<int N>
  float nnearestNeighborDistance(const Vec3f& center)
  {
    assert(!m_useHashCells);
    assert(photonCount() >= N);

    std::array<uint32_t, N> indices;
    std::array<float, N> outDistancesSq;
    tree.knnSearch(&center[0], indices.size(), &indices[0], &outDistancesSq[0]);
    std::sort(outDistancesSq.begin(), outDistancesSq.end());
    return std::sqrt(outDistancesSq[N-1]);
  }

  std::size_t photonCount() const
  {
    return photons.size();
  }

  std::size_t emittedCount() const
  {
    return m_emittedCount;
  }

  void increaseEmittedCount()
  {
    m_emittedCount += 1;
  }

private:
  std::size_t m_emittedCount;
  std::vector<Photon> photons;
  std::vector<Vec3f> points;

  bool m_useHashCells;
  PointHashCells hashCells;

  float adaptorConvertRadius(float r) { return square(r); } // L2 adaptor uses square in the current version but doesn't use the adaptor to fix it
  using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudf>, PointCloudf, 3 >;
  PointCloudf cloud = PointCloudf(points);
  my_kd_tree_t tree;
  std::vector<std::pair<uint32_t, float> > ret_matches;
};

//https://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/zackw/photon_mapping/PhotonMapping.html

// goal: make intent of stored paths clear
// use the Heckbert  syntax; L, D, S, E
// only store on first diffuse surface
// photon tracing: LS*D
// gathering : photon map  => LS*D (S|D)*E, opposite for irradiance estimate: 


void tracePhoton(RenderContext context, PhotonMap& photonMap, const Ray& a_ray, const Vec3f& a_flux, bool causticsMap)
{
  if (!a_flux.isFinite())
  {
    std::cout << " warning INPUT flux is not finite " << std::endl;
  }
  Vec3f flux = a_flux;
  LightRayInfo lightRay;

  Ray ray = a_ray;
  RayHit rayhit;
  context.scene.rayHit(ray, &rayhit);
  if (!hasHitSurface(rayhit)) return;

  Vec3f transmittance = beersLaw(lightRay.getTransmittance(), (rayhit.interaction.x - ray.origin()).norm());
  flux = flux.cwiseProd(transmittance);

  bool storeOnSpecular = false;

  for (int depth = 0; depth < context.renderSettings.maxDepth; depth++)
  {
    if (!flux.isFinite())
    {
      std::cout << " warning flux is not finite " << std::endl;
      break;
    }
    // now we are at hit.x, store photon choose new direction and use brdf
    bool doStore;
    if(causticsMap) doStore = context.scene.hasBRDFDiffuseLobe(rayhit) && depth > 0; // don't store first object, thats not a caustic mate
    else doStore = storeOnSpecular || context.scene.hasBRDFDiffuseLobe(rayhit);
    if (doStore)
    {
      photonMap.addPhoton(ray, rayhit, flux);
    }

    // sample output direction
    bool adjoint = true;
    SampleResult sampleResult;
    if (causticsMap)
    {
      sampleResult = context.scene.sampleBRDFSpecular(context, lightRay, rayhit, adjoint);
    }
    else
    {
      sampleResult = context.scene.sampleBRDF(context, lightRay, rayhit, adjoint);
    }
    lightRay = sampleResult.lightRay;
    Vec3f throughputThis = sampleResult.throughput();
    assertFinite(throughputThis);
    lightRay.applyWavelength(throughputThis);
    if (throughputThis == Vec3f::Zero()) break;

    Ray ray2 = context.scene.constructRay(rayhit.interaction, sampleResult.direction, sampleResult.outside);
    RayHit rayhit2;
    context.scene.rayHit(ray2, &rayhit2);
    if (!hasHitSurface(rayhit2) || isSelfIntersection(rayhit, rayhit2)) break;

    //beer's law; moving from hit.x to hit2.x;
    lightRay.updateMedia(context, rayhit, sampleResult.direction);
    Vec3f transmittance = beersLaw(lightRay.getTransmittance(), (rayhit2.interaction.x - ray2.origin()).norm());
    throughputThis = throughputThis.cwiseProd(transmittance);

    // should we absorb photon:
    float minQ = 1.0f - std::pow(0.1f, 1.0f / context.renderSettings.maxDepth); // probability roulette after maxdepth, 1.0 - p(r0 | r1 | r2 ..) = 1.0 - p(r_i)^80 = 1.0 - pr^80 = 0.9 , pr^80 =  0.1
    float rouletteQ = std::max(minQ, 1.0f - throughputThis.norm()); // trying to keep the flux the same....  
    if (context.rng.uniform01f() <= rouletteQ)
    {
      break;
    }

    flux = flux.cwiseProd(throughputThis) / (1.0f - rouletteQ);
    assertFinite(flux);

    rayhit = rayhit2;
    ray = ray2;
  }
}
struct PhotonFluxEstimate
{
  Vec3f fluxEstimate = Vec3f::Zero();
  int M;
};
PhotonFluxEstimate photonMapFluxEstimate(RenderContext context, PhotonMap &map, PhotonSearchResult &photons, const Ray& ray, const RayHit& rayhit, float &radius) {
  PhotonFluxEstimate result;

  if (radius <= 0.0f)
  {
    radius = 1.01f * map.nnearestNeighborDistance<5>(rayhit.interaction.x);
  }

  map.radiusSearch(rayhit.interaction.x, radius, &photons);

  bool adjoint = false;

  for (int i = 0; i < photons.size(); i++)
  {
    const Photon& photon = photons[i];
    MaterialEvalResult evalResult = context.scene.evaluteBRDF(rayhit, rayhit.wFrom(), photon.wi, 1.0f, adjoint, ShaderEvalFlag::DIFFUSE);
    assertFinite(evalResult.fDiffuse);
    assertFinite(evalResult.fSpecular);
    assertFinite(photon.flux);
    float fluxCorrection = std::abs(photon.wi.dot(rayhit.interaction.normalGeometric)) / (1e-7f + std::abs(photon.ngwi));
    result.fluxEstimate += (photon.flux * fluxCorrection).cwiseProd(evalResult.fDiffuse + evalResult.fSpecular);
  }
  result.M = photons.size();
  return result;
};
struct GatherPhotonsResult
{
  Vec3f photonFlux = Vec3f::Zero();
  Vec3f L = Vec3f::Zero();
  int M = 0;
};
GatherPhotonsResult gatherPhotons(RenderContext context, PhotonMap& photonMap, const Ray& a_ray, const RayHit& a_rayhit, Vec3f& a_normal, Vec3f& albedo, float &radius)
{
  GatherPhotonsResult result;

  LightRayInfo lightRay;

  if (!hasHitSurface(a_rayhit))
  {
    Vec3f transmittance = beersLaw(lightRay.getTransmittance(), 1e6f);
    
    Vec3f Le = context.scene.evalEnvironment(a_ray);
    result.L = transmittance.cwiseProd(Le);
    a_normal = Vec3f::Zero();
    albedo = result.L;
    return result;
  }
  a_normal = a_rayhit.interaction.normalGeometric;
  albedo = context.scene.getAlbedo(a_rayhit.interaction);

  //doesn't include emission
  PhotonSearchResult photons;

  RayHit rayhit = a_rayhit;
  Ray ray = a_ray;
  Vec3f throughput = Vec3f::Ones();
  //first we need to find first diffuse surface: TODO try SECOND diffuse surface
  for (int depth = 0; depth < 80; depth++)
  {
    //beer's law; moving from hit.x to hit2.x;
    Vec3f transmittance = beersLaw(lightRay.getTransmittance(), (rayhit.interaction.x - ray.origin()).norm());
    throughput = throughput.cwiseProd(transmittance);

    float rouletteQ = 0.0f;
    if (depth >= context.renderSettings.roulleteStartDepth) rouletteQ = context.renderSettings.rouletteQ;

    // https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting#
    if (context.rng.uniform01f() <= rouletteQ)
    {
      break;
    }
    throughput /= (1.0f - rouletteQ);

    result.L += throughput.cwiseProd(context.scene.getEmissionRadiance(rayhit.wFrom(), rayhit));

    bool doADiffuseBounce = false;
    bool adjoint = false;
    SampleResult sampleResult;
    if (doADiffuseBounce && depth == 0)
    {
      sampleResult = context.scene.sampleBRDF(context, lightRay, rayhit, adjoint);
    }
    else
    {
      sampleResult = context.scene.sampleBRDFSpecular(context, lightRay, rayhit, adjoint);

      if (context.scene.hasBRDFDiffuseLobe(rayhit))
      {
        PhotonFluxEstimate estimate = photonMapFluxEstimate(context, photonMap, photons, ray, rayhit, radius);
        result.M += estimate.M;
        result.photonFlux += throughput.cwiseProd(estimate.fluxEstimate);
      }
    }
    Vec3f throughputThis = sampleResult.throughput();
    assertFinite(throughputThis);

    lightRay.applyWavelength(throughput);

    throughput = throughput.cwiseProd(throughputThis);
    if (throughput == Vec3f::Zero()) break;

    Ray ray2 = context.scene.constructRay(rayhit.interaction, sampleResult.direction, sampleResult.outside);
    RayHit rayhit2;
    context.scene.rayHit(ray2, &rayhit2);

    lightRay.updateMedia(context, rayhit, ray2.direction());

    if (isSelfIntersection(rayhit, rayhit2)) break;

    if (!hasHitSurface(rayhit2))
    {
      Vec3f Le = context.scene.evalEnvironment(ray2);
      result.L += throughput.cwiseProd(Le);
      break;
    }

    rayhit = rayhit2;
    ray = ray2;
  }
  
  return result;
}

// caustics map builds paths according to L S+ D, to not account for these paths twice, we have to return the flux and irradiance in a way where we can tell whether the energy came through a caustic path or not
struct PathTraceWithCausticsMapResult
{
  Vec3f photonFlux = Vec3f::Zero();
  Vec3f LLight = Vec3f::Zero();
  Vec3f LRest = Vec3f::Zero();
  Vec3f LSpecular = Vec3f::Zero();
  Vec3f LCaustic = Vec3f::Zero();
  int M = 0;
  bool specularLightPath = false;
};
PathTraceWithCausticsMapResult pathTraceWithCausticsMap(RenderContext context, PhotonMap& photonMap, LightRayInfo& lightRay, const Ray& ray, RayHit *rayhitPtr, Vec3f& normal, Vec3f& albedo, float &radius, int depth)
{
  PathTraceWithCausticsMapResult result;

  if (depth >= context.renderSettings.maxDepth) return result;
  
  RayHit rayhitLocal;
  RayHit& rayhit = rayhitPtr == nullptr ? rayhitLocal : *rayhitPtr;
  if (rayhitPtr == nullptr) context.scene.rayHit(ray, &rayhit);

  PhotonSearchResult photons;

  if (hasHitSurface(rayhit))
  {
    if (depth == 0)
    {
      normal = rayhit.interaction.normalShadingDefault;
      albedo = context.scene.getAlbedo(rayhit.interaction);
    }

    Vec3f throughput = beersLaw(lightRay.getTransmittance(), (rayhit.interaction.x - ray.origin()).norm());
    
    if (context.scene.isSurfaceLight(rayhit))
    {
      result.LLight += throughput.cwiseProd(context.scene.getEmissionRadiance(-ray.direction(), rayhit));
      return result;
    }

    if (context.scene.hasBRDFDiffuseLobe(rayhit))
    {
      PhotonFluxEstimate estimate = photonMapFluxEstimate(context, photonMap, photons, ray, rayhit, radius);
      result.M += estimate.M;
      result.photonFlux += throughput.cwiseProd(estimate.fluxEstimate);
    }

    constexpr int beta = 2;
    SamplingStrategies strategies(context);
    int numStrategies = strategies.numStrategies();
    SmallVector<float, 3> pdfs(numStrategies);

    // https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting#
    float rouletteQ = 0.0f;
    if (depth >= context.renderSettings.roulleteStartDepth) rouletteQ = context.renderSettings.rouletteQ;

    if (context.rng.uniform01f() <= rouletteQ)
    {
      return result;
    }
    throughput /= (1.0f - rouletteQ);

    Vec3f estimator = Vec3f::Zero();

    for (int strategy = 0; strategy < numStrategies; strategy++)
    {
      MISSampleResult sampleResult = strategies.sample(context, lightRay, ray, rayhit, strategy);
      Vec3f throughputWavelength = Vec3f::Ones();
      sampleResult.lightRay.applyWavelength(throughputWavelength);

      pdfs[strategy] = sampleResult.pdfOmega;

      if (sampleResult.throughput() == Vec3f::Zero() || isSelfIntersection(rayhit, sampleResult.rayhit)) continue;

      for (int j = 0; j < numStrategies; j++)
      {
        if (j != strategy) pdfs[j] = strategies.evalPDF(context, rayhit, -ray.direction(), j, sampleResult.ray, sampleResult.rayhit);
      }
      float msiWeight = computeMsiWeightPowerHeuristic<beta>(strategy, pdfs);
      
      sampleResult.lightRay.updateMedia(context, rayhit, sampleResult.ray.direction());
      
      assertFinite(throughput);
      assertFinite(sampleResult.throughputDiffuse);
      assertFinite(sampleResult.throughputSpecular);
      Vec3f throughputDiffuseT = throughput.cwiseProd(sampleResult.throughputDiffuse.cwiseProd(throughputWavelength) * msiWeight);
      Vec3f throughputSpecularT = throughput.cwiseProd(sampleResult.throughputSpecular.cwiseProd(throughputWavelength) * msiWeight);
      Vec3f throughput2 = throughputDiffuseT + throughputSpecularT;
      assertFinite(throughput2);

      Vec3f normal2, albedo2;
      PathTraceWithCausticsMapResult rs = pathTraceWithCausticsMap(context, photonMap, sampleResult.lightRay, sampleResult.ray, &sampleResult.rayhit, normal2, albedo2, radius, depth + 1);

      result.LSpecular += (rs.LLight + rs.LSpecular).cwiseProd(throughputSpecularT);
      result.LCaustic += rs.LSpecular.cwiseProd(throughputDiffuseT);
      //result.LRest += (rs.LRest + rs.LCaustic).cwiseProd(throughput2) + rs.LLight.cwiseProd(throughputDiffuseT);
      result.LRest += (rs.LRest ).cwiseProd(throughput2) + rs.LLight.cwiseProd(throughputDiffuseT);
      result.photonFlux += rs.photonFlux.cwiseProd(throughput2);
      result.M += rs.M;
    }

    return result;
  }
  else
  {
    Vec3f transmittance = beersLaw(lightRay.getTransmittance(), 1e6f);
    Vec3f Le = context.scene.evalEnvironment(ray);
    result.LRest = transmittance.cwiseProd(Le);
    normal = Vec3f::Zero();
    albedo = result.LRest;
    return result;
  }
}
float computeNewRadius(float radius1, int sampleIdx)
{
  float alpha = 0.5f;
  float scale = 1.0f;
  for (int j = 0; j < sampleIdx; j++)
  {
    scale *= (j + alpha) / (j + 1.0f);
  }
  return radius1 * sqrt(scale);
}

// trying to get cost based estimate
// what is the cost: #Point test + #Cell tests
//  #Cell tests ~ \sum_i (radius_i / cellSize)^3
//  #Point tests ~ \sum_i  max(radius_i, cellSize)^3 * density
// min a \sum_i (radius_i / cellSize)^3 + \sum_i  max(radius_i, cellSize)^3 * density
// min sum_i (radius_i / cellSize)^3 + \sum_i  max(radius_i, cellSize)^3
//
float computeCellSize(const Image1f& radii)
{
  float sum = 0.0f;
  for (uint32_t i = 0; i < radii.getWidth(); i++)
  {
    for (uint32_t j = 0; j < radii.getHeight(); j++)
    {
      sum += radii(i, j);
    }
  }
  float a = 5.0f;
  float density = 1e5;
  float cellSize = sum / (radii.getWidth() * radii.getHeight());
  auto computeCost = [&radii, density, a](float cs) {

    float sum = 0.0f;
    for (uint32_t i = 0; i < radii.getWidth(); i++)
    {
      for (uint32_t j = 0; j < radii.getHeight(); j++)
      {
        sum += a * powci<3>(radii(i, j) / cs) + powci<3>(std::max(radii(i, j), cs)) * density;
      }
    }
    return sum;
  };
  float cost = computeCost(cellSize);
  for (int j = 0; j < 10; j++)
  {
    bool changed = false;
    float trial = cellSize * 1.5f;
    float trialCost = computeCost(trial);
    if (trialCost < cost)
    {
      cellSize = trial;
      cost = trialCost;
      changed = true;
    }
    if (!changed) break;
  }
  for (int j = 0; j < 10; j++)
  {
    bool changed = false;
    float trial = cellSize * 0.6666f;
    float trialCost = computeCost(trial);
    if (trialCost < cost)
    {
      cellSize = trial;
      cost = trialCost;
      changed = true;
    }
    if (!changed) break;
  }
  return cellSize;
}

float computeRadiusPixelFootPrint(RenderContext context, float defaultR, int i, int j, int width, int height, bool ignoreCosineTerm)
{
  assert(ignoreCosineTerm);
  
  Vec2f p0(i, j);
  Vec2f offset(0.5f, 0.5f);
  Ray ray = context.scene.camera.ray(context.rng, p0 + offset, float(width), float(height));
  RayHit rayhit;
  context.scene.rayHit(ray, &rayhit);
  if (!hasHitSurface(rayhit))
  {
    return defaultR;
  }
  float angle = context.scene.camera.rayAngle(p0 + offset, float(width), float(height));
  float theta = std::abs(std::acos(-ray.direction().dot(rayhit.interaction.normalGeometric)));
  float d = (context.scene.camera.position() - rayhit.interaction.x).norm();

  if (!context.scene.hasBRDFDiffuseLobe(rayhit)) // if we hit purely specular surface, we should track down a diffuse surface...
  {
    for (int i = 0; i < context.renderSettings.maxDepth; i++)
    {
      if (context.scene.hasBRDFDiffuseLobe(rayhit)) // if we hit purely specular surface, we should track down a diffuse surface...
      {
        break;
      }
      bool adjoint = false;

      SampleResult sampleResult;

      int tries = 0;
      do
      {
        sampleResult = context.scene.sampleBRDFSpecular(context, LightRayInfo(), rayhit, adjoint);
        tries += 1;
      } while (sampleResult.throughputSpecular == Vec3f::Zero() && tries < 3);
      if (sampleResult.throughputSpecular == Vec3f::Zero()) break;

      Ray ray2 = context.scene.constructRay(rayhit.interaction, sampleResult.direction, sampleResult.outside);
      RayHit rayhit2;
      context.scene.rayHit(ray2, &rayhit2);
      
      if (!hasHitSurface(rayhit2) || isSelfIntersection(rayhit, rayhit2))
      {
        return defaultR;
      }

      d += (rayhit.interaction.x - rayhit2.interaction.x).norm();

      rayhit = rayhit2;
    }
  }

  //if (ignoreCosineTerm)
  {
    //          ////////////////
    //  /  angl   r
    //  j-------------------- d |
    // r / d = tan(angle)
    float radius = 0.5f * d * std::tan(angle);
    return radius;
  }
  ////               q             ---                     / 
  ////              ----angle     | theta   / radius
  ////  -------------------------   /  normal
  ////              d
  ////  radius*radius = q*q +  d*d - 2*q * d * cos(angle)
  ////  q * q = d*d + radius * radius - 2 * d * radius * cos( theta )
  //// thirdAngle = pi - angle - theta
  //// law of sines:  d/sin(thirdAngle) = q / sin(theta)
  //// law of sines:  radius/sin(angle) = q / sin(thirdAngle)
  //float thirdAngle = float(Pi) - angle - (0.5f * float(Pi) + theta);
  //float radius1 = std::sin(angle) * d / std::sin(thirdAngle);
  //return 0.5f * radius1;
}

float getCloseNonNegative(const Image1f& image, int i, int j)
{
  int k = 1;
  while (k < std::max(image.getHeight(), image.getWidth()))
  {
    for (int m = 0; m <= k * 2; m++)
    {
      int i2, j2;
      i2 = i - k;
      j2 = j - k + m;
      if (image.inside(i2, j2) && image(i2, j2) > 0.0f) return image(i2, j2);
      i2 = i - k + m;
      j2 = j - k;
      if (image.inside(i2, j2) && image(i2, j2) > 0.0f) return image(i2, j2);
      i2 = i + k;
      j2 = j - k + m;
      if (image.inside(i2, j2) && image(i2, j2) > 0.0f) return image(i2, j2);
      i2 = i - k + m;
      j2 = j + k;
    }

    k += 1;
  }

  return 1e-3f;
}

Image1f computeInitialRadii(RenderContext context, std::size_t width, std::size_t height, float footprintScale)
{
  Image1f radii;
  radii.setConstant(width, height, -1.0f);

  for (uint32_t j = 0; j < height; j++)
  {
    for (uint32_t i = 0; i < width; i++)
    {
      bool ignoreCosineTerm = true;
      radii(i, j) = footprintScale * computeRadiusPixelFootPrint(context, -1.0f, i, j, width, height, ignoreCosineTerm);
    }
  }
  for (uint32_t j = 0; j < height; j++)
  {
    for (uint32_t i = 0; i < width; i++)
    {
      if (radii(i, j) < 0.0f)
      {
        float r = getCloseNonNegative(radii, i, j);
        radii(i, j) = r;
      }
    }
  }
  return radii;
}

Image1f computeInitialRadiiTrace(RenderContext context, PMRenderJob& job, PhotonMap &photonMap, int tracethreshold, bool causticsMap)
{
  Image1f radii;
  radii.setConstant(job.renderSettings.width, job.renderSettings.height, -1.0f);
  Scene& scene = *job.scene;
  
  photonMap.clear(false, -1.0f);

  std::cout << " tracePhoton for radius " << std::endl;
  for (int i = 0; i <  tracethreshold && photonMap.photonCount() < tracethreshold; i++)
  {
    Vec3f flux;
    Ray ray = scene.sampleLightRay(context.rng, &flux);
    photonMap.increaseEmittedCount();
    tracePhoton(context, photonMap, ray, flux, causticsMap);
  }
  photonMap.build();
  
  constexpr int NeighborCount = 25;
  if (photonMap.photonCount() < NeighborCount)
  {
    for (uint32_t j = 0; j < job.renderSettings.height; j++)
    {
      for (uint32_t i = 0; i < job.renderSettings.width; i++)
      {
        bool ignoreCosineTerm = true;
        float footprint = computeRadiusPixelFootPrint(context, -1.0f, i, j, job.renderSettings.width, job.renderSettings.height, ignoreCosineTerm);
        radii(i, j) = footprint;
      }
    }
  }
  else
  {
    for (uint32_t j = 0; j < job.renderSettings.height; j++)
    {
      for (uint32_t i = 0; i < job.renderSettings.width; i++)
      {
        Vec2f p0(i, j);
        Vec2f offset(0.5f, 0.5f);
        Ray ray = context.scene.camera.ray(context.rng, p0 + offset, float(job.renderSettings.width), float(job.renderSettings.height));
        RayHit rayhit;
        context.scene.rayHit(ray, &rayhit);
        if (!hasHitSurface(rayhit))
        {
          continue;
        }
        
        bool ignoreCosineTerm = true;
        float footprint = computeRadiusPixelFootPrint(context, -1.0f, i, j, job.renderSettings.width, job.renderSettings.height, ignoreCosineTerm);
        radii(i, j) = std::min(50.0f * footprint,  std::max( footprint, photonMap.nnearestNeighborDistance<NeighborCount>(rayhit.interaction.x)) );
      }
    }
  }
  for (uint32_t j = 0; j < job.renderSettings.height; j++)
  {
    for (uint32_t i = 0; i < job.renderSettings.width; i++)
    {
      if (radii(i, j) < 0.0f)
      {
        float r = getCloseNonNegative(radii, i, j);
        radii(i, j) = r;
      }
    }
  }
  return radii;
}

float computeAvg(const Image1f& r)
{
  float sum = 0.0f;
  for (int j = 0; j < r.getHeight(); j++)
  {
    for (int i = 0; i < r.getWidth(); i++)
    {
      sum += r(i, j);
    }
  }
  float avg = sum / (r.getWidth() * r.getHeight());
  return avg;
}

void PMRenderJob::loadJSON(std::filesystem::path jsonFilepath)
{
  std::filesystem::path folder = jsonFilepath.parent_path();
  nlohmann::json jsonData = readJSON(jsonFilepath);
  from_json(jsonData["settings"], renderSettings);
  std::filesystem::path scenePath = folder / jsonData["scene"].get<std::string>();
  this->scene = std::make_unique<Scene>(scenePath);

  pixels.clear();
  for (int i = 0; i < renderSettings.width * renderSettings.height; i++) pixels.emplace_back();
  pixelsWidth = renderSettings.width;
}


//Stochastic progressive photon mapper
// instead of tau we store L
// but first we substitute tau/(\pi r^2) = \Lambda
// changed equations
// Eq. 7 : L_i(S, omega) = \Lambda_i / (N_e(i) )
// Eq. 11: \Lambda_{i+1} \pi r_{i+1}^2 = (\Lambda_{i} \pi r_i^2 + \Phi_i) r_{i+1}^2/r_{i}^2   =>  \Lambda_{i+1} = \Lambda_i + \flux_i / (\pi r_{i}^2)
// 
// L_i(S, omega) = \Lambda_i / (N_e(i) )
// L_{i+1}(S, omega) = \Lambda_{i+1} / N_e(i+1) = (\Lambda_i + \flux_i / (\pi r_{i}^2)) / N_e(i+1)
//    = (\Lambda_i + \flux_i / (\pi r_{i}^2)) / N_e(i+1) = \Lambda_i / N_e(i+1) + \flux_i / (\pi r_{i}^2)  N_e(i+1) )
//    = L_{i}(S) N_e(i) / N_e(i+1) + \flux_i / (\pi r_{i}^2)  N_e(i+1) )
// 
void renderPhotonThread(Image3fAccDoubleBuffer& colorBuffer, Image3fAccDoubleBuffer& normalBuffer, Image3fAccDoubleBuffer& albedoBuffer, PMRenderJob& job, RenderThreadData& threadData)
{
  preinitEmbree();

  Scene& scene = *job.scene;

  RNG rng(threadData.seed);

  Image3f& imageColor = colorBuffer.getWriteBuffer().data;
  imageColor.setZero(job.renderSettings.width, job.renderSettings.height);
  Image3f& imageNormal = normalBuffer.getWriteBuffer().data;
  imageNormal.setZero(job.renderSettings.width, job.renderSettings.height);
  Image3f& imageAlbedo = albedoBuffer.getWriteBuffer().data;
  imageAlbedo.setZero(job.renderSettings.width, job.renderSettings.height);
  // 
  RenderContext context(scene, rng, job.renderSettings);

  PhotonMap photonMap;
  bool useCausticsMap = true;
  int tracethreshold = job.renderSettings.height * job.renderSettings.width;
  int countThreshold = 0.5f * job.renderSettings.height * job.renderSettings.width;
  if(useCausticsMap) tracethreshold = 10.f * job.renderSettings.height * job.renderSettings.width; // using higher trace threshold is good when the count treshold isn't reached at all

  std::cout << " radii begin " << std::endl;
  Image1f radii;
  radii.setConstant(job.renderSettings.width, job.renderSettings.height, 1.f);
  if(scene.hasSurfaceLight()) radii = computeInitialRadii(context, job.renderSettings.width, job.renderSettings.height, 10.0f);
  //Image1f radii;
  //radii.setConstant(job.renderSettings.width, job.renderSettings.height, -1.f);
  //Image1f radii = computeInitialRadiiTrace(context, job, photonMap, tracethreshold, useCausticsMap);

  std::cout << " end " << std::endl;

  RenderJobSample sample = job.fetchSample();
  while (!job.stopFlag && sample.idx < job.renderSettings.numSamples)
  {
    std::cout << " sampleIdx " << sample.idx << " numSamples " << colorBuffer.getWriteBuffer().numSamples << '\n';

    float cellSize = computeCellSize(radii);
    std::cout << " cellSize " << cellSize << " avg radius " << computeAvg(radii) << std::endl;
    HighResTimer timer;
    photonMap.clear(true, cellSize);

    if (scene.hasSurfaceLight())
    {
      std::cout << " tracePhoton " << std::endl;
      Vec3f totalFlux = Vec3f::Zero();
      for (int i = 0; i < tracethreshold && photonMap.photonCount() < countThreshold; i++)
      {
        Vec3f flux;
        Ray ray = scene.sampleLightRay(rng, &flux);
        totalFlux += flux;
        photonMap.increaseEmittedCount();
        tracePhoton(context, photonMap, ray, flux, true);
      }
      std::cout << " totalFlux " << totalFlux / photonMap.emittedCount() << std::endl;
      std::cout << " building, emitted " << photonMap.emittedCount() << std::endl;
    }
    photonMap.build();
    std::cout << " photonmap size " << photonMap.photonCount() << std::endl;

    HighResTimer timer2;
    for (uint32_t j = 0; j < job.renderSettings.height; j++)
    {
      for (uint32_t i = 0; i < job.renderSettings.width; i++)
      {
        Vec2f p0 = Vec2f(float(i), float(j));
        Ray ray = scene.camera.ray(rng, p0 + sample.offset, float(job.renderSettings.width), float(job.renderSettings.height));
        float radius = job.lock(i, j, radii(i, j));
        Vec3f normal, albedo;
        LightRayInfo lightRay;

        PathTraceWithCausticsMapResult photonsResult = pathTraceWithCausticsMap(context, photonMap, lightRay, ray, nullptr, normal, albedo, radius, 0);
        if (photonsResult.photonFlux != photonsResult.photonFlux)
        {
          std::cout << " error: flux is NAN " << i << ' ' << j << std::endl;
        }
        Vec3f L = photonsResult.LLight + photonsResult.LRest + photonsResult.LSpecular;
        if (L != L)
        {
          std::cout << " error: L is NAN " << i << ' ' << j << std::endl;
        }

        assertFinite(photonsResult.photonFlux);
        Vec3f Lnew;
        if (L != L || photonsResult.photonFlux != photonsResult.photonFlux)
        {
          Lnew = job.unlock(i, j);
        }
        else
        {
          Lnew = job.unlockAndUpdateStatistic(i, j, photonsResult.photonFlux, L, photonsResult.M, photonMap.emittedCount(), &radius);
        }
        assertFinite(Lnew);
        assertFinite(normal);
        assertFinite(albedo);
        radii(i, j) = radius;
        imageColor(i, j) = Lnew;
        imageNormal(i, j) = normal;
        imageAlbedo(i, j) = albedo;
      }
    }

    photonMap.printStats();
    std::cout << "gather time " << timer2.seconds() << std::endl;

    colorBuffer.getWriteBuffer().numSamples = 1;
    normalBuffer.getWriteBuffer().numSamples = 1;
    albedoBuffer.getWriteBuffer().numSamples = 1;
    colorBuffer.copyBuffer();
    normalBuffer.copyBuffer();
    albedoBuffer.copyBuffer();

    sample = job.fetchSample();

    std::cout << (1 / timer.seconds()) << "samples per second" << std::endl;
  }
  std::cout << " stopping " << '\n';
  threadData.stoppedFlag = true;
}

}
