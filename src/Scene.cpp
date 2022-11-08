#include <Hasty/Scene.h>

#include <Hasty/PathTracing.h>
#include <Hasty/Texture.h>
#include <Hasty/Json.h>

#include <numeric>

namespace Hasty
{

Scene::Scene(std::filesystem::path inputfilePath)
{
  std::filesystem::path sceneFolder = inputfilePath.parent_path();
  nlohmann::json jsonData = readJSON(inputfilePath);

  auto jsonCamera = jsonData.at("camera");
  this->camera = jsonCamera.get<Camera>();
  if (jsonData.at("world").contains("background_color"))
  {
    this->background = std::make_unique<BackgroundColor>(jsonData.at("world").at("background_color").get<Vec3f>());
  }
  else
  {
    std::string envFilename = jsonData.at("world").at("environment_texture").get<std::string>();
    this->background = std::make_unique<EnvironmentTexture>(sceneFolder / envFilename);
  }
  bool varyGlassIOR;
  json_get_optional(varyGlassIOR, jsonData.at("world"), "vary_glass_ior", false);

  std::filesystem::path meshFilePath = sceneFolder / jsonData.at("mesh").get<std::string>();
  tinyobj::ObjReaderConfig reader_config;
  reader_config.mtl_search_path = sceneFolder.string();
  //reader_config.triangulate = false; //got burned too many times by the automatic triangulation

  if (!reader.ParseFromFile(meshFilePath.string(), reader_config))
  {
    if (!reader.Error().empty())
    {
      throw std::runtime_error("TinyObjReader error " + reader.Error());
    }
    else
    {
      throw std::runtime_error("unknown error tinyobjreader");
    }
  }

  if (!reader.Warning().empty())
  {
    std::cout << "TinyObjReader: " << reader.Warning();
  }

  const tinyobj::attrib_t& attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();
  const std::vector<tinyobj::material_t>& materials = reader.GetMaterials();

  std::cout << " shapes " << shapes.size() << std::endl;
  std::cout << " materials " << materials.size() << std::endl;

  m_device = rtcNewDevice(NULL);
  m_embreeScene = rtcNewScene(m_device);

  //convert materials to our brdfs

  m_brdfs.clear();
  for (const tinyobj::material_t& material : materials)
  {
    Vec3f materialDiffuse(material.diffuse[0], material.diffuse[1], material.diffuse[2]);

    if (material.dissolve == 0.0f)
    {
      m_brdfs.emplace_back(std::make_unique<GlassBTDF>(materialDiffuse, material.ior, varyGlassIOR));
    }
    else
    {
      
      std::unique_ptr<ITextureMap3f> albedoTexture;
      if (material.diffuse_texname != "")
      {
        albedoTexture = Texture3f::loadTexture((sceneFolder / material.diffuse_texname).string(), ColorSpace::sRGB);
      }
      else
      {
        albedoTexture = std::make_unique<ConstantTexture3f>(materialDiffuse);
      }
      
      std::unique_ptr<ITextureMap1f> roughnessTexture;
      if (material.specular_highlight_texname != "") // the naming between the blender exporter and tinyobj is not consistent
      {
        roughnessTexture = Texture1f::loadTexture((sceneFolder / material.specular_highlight_texname).string(), ColorSpace::Linear);
      }
      else
      {
        float nSpecular = material.shininess;
        float roughness = 1.0f - std::sqrt(nSpecular / 1000.0f); // blender currently uses the formula (1.0 - bsdf_roughness)^2 * 1000
        roughnessTexture = std::make_unique<ConstantTexture1f>(roughness);
      }

      std::unique_ptr<ITextureMap1f> metallicTexture;
      if (material.metallic_texname != "")
      {
        metallicTexture = Texture1f::loadTexture((sceneFolder / material.metallic_texname).string(), ColorSpace::Linear);
      }
      else
      {
        float metallic = material.ambient[0];
        if (metallic == 1.0f)
        {
          std::cout << " warning default blender exporter exports metallic = 1, when in fact it is set to 0, consider modifying Blender/scripts/addons/io_scene_obj/export_obj.py, note that there are currently two obj exporters in blender so be careful " << std::endl;
        }
        metallicTexture = std::make_unique<ConstantTexture1f>(metallic);
      }

      m_brdfs.emplace_back(std::make_unique<PrincipledBRDF>(std::move(albedoTexture), std::move(roughnessTexture) , std::move(metallicTexture), material.specular[0], material.ior));
    }
  }

  std::vector<float> lightAreas;

  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++)
  {
    auto& shape = shapes[s];

    RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vertexBuffer = (float*)rtcSetNewGeometryBuffer(geom,
      RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), attrib.vertices.size() / 3); // instead of remapping vertices we just  copy vertex buffer for each subobject for now

    std::size_t numFaces = shape.mesh.num_face_vertices.size();
    unsigned* indexBuffer = (unsigned*)rtcSetNewGeometryBuffer(geom,
      RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), numFaces);

    for (int vertexIndex = 0; vertexIndex < attrib.vertices.size(); vertexIndex++)
    {
      vertexBuffer[vertexIndex] = attrib.vertices[vertexIndex];
    }

    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < numFaces; f++)
    {
      size_t fv = shape.mesh.num_face_vertices[f];
      if (fv != 3)
      {
        throw std::runtime_error(" dont support more than 3 points per face ");
      }

      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++)
      {
        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

        indexBuffer[index_offset + v] = idx.vertex_index;

      }
      index_offset += fv;


      int materialId = shape.mesh.material_ids[f];
      if (materialId == -1)
      {
        throw std::runtime_error(" error face has no material ");
      }
      auto& mat = materials[materialId];
      bool isLight = mat.emission[0] > 0.0f || mat.emission[1] > 0.0f || mat.emission[2] > 0.0f;
      if (isLight)
      {
        if (mat.diffuse[0] > 0.0f || mat.diffuse[1] > 0.0f || mat.diffuse[2] > 0.0f
          || mat.specular[0] > 0.0f || mat.specular[1] > 0.0f || mat.specular[2] > 0.0f)
        {
          throw std::runtime_error(" don't support non black diffuse materials with emission ");
        }
        std::array<std::size_t, 2> ids = { s, f };
        lightTriangles.emplace_back(ids);

        std::array<Vec3f, 3> p = collectTriangle(s, f);
        float area = 0.5f * (p[1] - p[0]).cross(p[2] - p[0]).norm();

        lightAreas.emplace_back(area);
      }
    }


    rtcCommitGeometry(geom);
    rtcAttachGeometry(m_embreeScene, geom);
    rtcReleaseGeometry(geom);
  }

  rtcCommitScene(m_embreeScene);

  lightDistribution = std::make_unique<std::discrete_distribution<uint32_t> >(begin(lightAreas), end(lightAreas));
  totalLightArea = std::accumulate(begin(lightAreas), end(lightAreas), 0.f);
}
Scene::~Scene()
{
  rtcReleaseScene(m_embreeScene);
  rtcReleaseDevice(m_device);
}

SurfaceInteraction Scene::getInteraction(unsigned int geomID, unsigned int primID, const Vec3f& barycentricCoordinates, const Vec3f& geomNormal)
{
  SurfaceInteraction interaction;

  const tinyobj::attrib_t& attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();
  
  std::array<Vec3f, 3> xv = collectTriangle(geomID, primID);

  interaction.geomID = geomID;
  interaction.primID = primID;
  interaction.barycentricCoordinates = barycentricCoordinates;
  interaction.x = barycentricCoordinates[0] * xv[0] + barycentricCoordinates[1] * xv[1] + barycentricCoordinates[2] * xv[2];
  interaction.normalGeometric = geomNormal;

  {
    bool failed = false;
    Vec2f uv = Vec2f::Zero();
    for (int v = 0; v < 3; v++)
    {
      tinyobj::index_t idx = shapes[geomID].mesh.indices[3 * primID + v];
      if (2 * size_t(idx.texcoord_index) + 1 >= attrib.texcoords.size())
      {
        failed = true;
        break;
      }
      Vec2f uv_j(attrib.texcoords[2 * size_t(idx.texcoord_index) + 0], attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]);

      uv += uv_j * barycentricCoordinates[v];
    }
    if (!failed)
    {
      interaction.uv = uv;
    }
  }

  interaction.materialId = shapes[geomID].mesh.material_ids[primID];

  bool interpolateNormals = shapes[geomID].mesh.smoothing_group_ids[primID] != 0;
  if (interpolateNormals)
  {
    std::array<Vec3f, 3> nv = collectTriangleNormals(geomID, primID);

    interaction.normalShadingDefault = barycentricCoordinates[0] * nv[0] + barycentricCoordinates[1] * nv[1] + barycentricCoordinates[2] * nv[2];
    interaction.normalShadingDefault.normalize();
    
    interaction.h = 0.0f;
    for (int i = 0; i < 3; i++)
    {
      // intersect plane defined by corner + corner normal with line through interaction.x and geometric normal to get some linear geometry which we then interpolate
      // (x + ng * t) . nv - xv . nv = 0 => 
      float nvng = interaction.normalGeometric.dot(nv[i]);
      float hi = nvng == 0.0f ? 1e6 : (xv[i].dot(nv[i]) - interaction.x.dot(nv[i])) / nvng;
      interaction.h += interaction.barycentricCoordinates[i] *  interaction.barycentricCoordinates[i] *  hi;
    }
  }
  else
  {
    interaction.normalShadingDefault = interaction.normalGeometric;
    interaction.h = 0.0f;
  }


  return interaction;
}

Vec3f Scene::constructRayX(const SurfaceInteraction& interaction, bool outside)
{
  // Shadow terminator issue: https://wiki.blender.org/wiki/Reference/Release_Notes/3.0/Cycles#Shadow_Terminator
  // we don't get smooth lighting if we dont take occlusion into account when sending rays towards e.g. the light source, the solution is: fake offset using some kind of offset based on a nonlinear patch based on the normals
  // possibilities: https://computergraphics.stackexchange.com/questions/10731/results-of-curved-pn-triangles-algorithm-has-visible-edges
  // https://developer.blender.org/rB9c6a382

  Vec3f normal = interaction.normalGeometric;

  float eps = 1e-5f;
  float hmin = std::max(eps, interaction.x.norm() * eps);
  Vec3f xOffset = interaction.x + normal * (1.01f * (outside ? std::max(0.0f, interaction.h) + hmin : std::min(0.0f, interaction.h) - hmin));
  return xOffset;
}
Ray Scene::constructRay(const SurfaceInteraction& interaction, const Vec3f& direction, bool outside)
{
  Vec3f xOffset = constructRayX(interaction, outside);
  return Ray(xOffset, direction);
}
Ray Scene::constructRayEnd(const SurfaceInteraction& interaction, const Vec3f& end, bool outside)
{
  Vec3f xOffset = constructRayX(interaction, outside);
  return Ray(xOffset, (end - xOffset).normalized());
}
void Scene::rayHit(const Ray& ray, RayHit* rayhit)
{
  RTCRayHit& rtcrayhit = rayhit->rtc;
  auto& orig = ray.origin();
  auto& dir = ray.direction();
  rtcrayhit.ray.org_x = orig[0]; rtcrayhit.ray.org_y = orig[1]; rtcrayhit.ray.org_z = orig[2];
  rtcrayhit.ray.dir_x = dir[0]; rtcrayhit.ray.dir_y = dir[1]; rtcrayhit.ray.dir_z = dir[2];
  rtcrayhit.ray.tnear = 1e-6f;
  rtcrayhit.ray.tfar = std::numeric_limits<float>::infinity();
  rtcrayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

  RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  rtcIntersect1(m_embreeScene, &context, &rtcrayhit);

  if (hasHitSurface(*rayhit))
  {
    Vec3f barycentricCoordinates(1.0f - rtcrayhit.hit.u - rtcrayhit.hit.v, rtcrayhit.hit.u, rtcrayhit.hit.v);
    Vec3f geomNormal = Vec3f(rtcrayhit.hit.Ng_x, rtcrayhit.hit.Ng_y, rtcrayhit.hit.Ng_z).normalized();
    rayhit->interaction = getInteraction(rayhit->rtc.hit.geomID, rayhit->rtc.hit.primID, barycentricCoordinates, geomNormal);
  }
}


BXDF& Scene::getBXDF(const SurfaceInteraction& interaction)
{
  return *m_brdfs[interaction.materialId];
}
const BXDF& Scene::getBXDF(const SurfaceInteraction& interaction) const
{
  return *m_brdfs[interaction.materialId];
}
MaterialEvalResult Scene::evaluteBRDF(const RayHit& hit, const Vec3f& wo, const Vec3f& wi, float indexOfRefractionOutside, bool adjoint, ShaderEvalFlag flags)
{
  return getBXDF(hit.interaction).evaluate(hit.interaction, wo, wi, indexOfRefractionOutside, adjoint, flags);
}

Vec3f Scene::sampleBRDF(RenderContext& context, LightRayInfo& lightRay, const RayHit& rayhit, bool adjoint, Vec3f* throughputDiffuse, Vec3f* throughputSpecular, float* pDensity, bool *outside)
{
  return getBXDF(rayhit.interaction).sample(context.rng, rayhit.interaction, lightRay, rayhit.wFrom(), [&lightRay, &context, &rayhit]() { return lightRay.getOutsideIOR(context, rayhit); }, adjoint, throughputDiffuse, throughputSpecular, pDensity, outside, ShaderEvalFlag::ALL);
}
Vec3f Scene::sampleBRDFSpecular(RenderContext& context, LightRayInfo& lightRay, const RayHit& rayhit, bool adjoint, Vec3f* throughputDiffuse, Vec3f* throughputSpecular, float* pDensity, bool *outside)
{
  return getBXDF(rayhit.interaction).sample(context.rng, rayhit.interaction, lightRay, rayhit.wFrom(), [&lightRay, &context, &rayhit]() { return lightRay.getOutsideIOR(context, rayhit); }, adjoint, throughputDiffuse, throughputSpecular, pDensity, outside, ShaderEvalFlag::SPECULAR);
}
float Scene::evaluateSamplePDF(const RayHit& rayhit, const Vec3f& direction2)
{
  return getBXDF(rayhit.interaction).evaluateSamplePDF(rayhit.interaction, rayhit.wFrom(), direction2);
}
bool Scene::hasBRDFDiffuseLobe(const RayHit& hit)
{
  return Scene::getBXDF(hit.interaction).hasDiffuseLobe(hit.interaction);
}
float Scene::getIORInside(const RayHit& rayhit, int wavelength)
{
  return Scene::getBXDF(rayhit.interaction).getIndexOfRefraction(wavelength);
}

const tinyobj::material_t& Scene::getMaterial(unsigned int geomId, unsigned int primId) const
{
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();
  int materialId = shapes[geomId].mesh.material_ids[primId];
  const std::vector<tinyobj::material_t>& materials = reader.GetMaterials();
  return materials[materialId];
}
const tinyobj::material_t& Scene::getMaterial(const RayHit& hit) const
{
  const std::vector<tinyobj::material_t>& materials = reader.GetMaterials();
  return materials[hit.interaction.materialId];
}

Vec3f Scene::getEmissionRadiance(const Vec3f& wo, unsigned int geomId, unsigned int primId) const
{
  auto& material = getMaterial(geomId, primId);
  Vec3f materialEmission(material.emission[0], material.emission[1], material.emission[2]);
  // materialEmission is in W / A, irradiance, whereas this function returns radiance
  // to derive it i will resort to are formulation: integrate over hemisphere with radius r:
  // \int_\Omega  L_E V(x, y) cos(theta_x) cos(theta_y)/ r^2 d\omega
  //     = \int_\Omega L_E cos(theta_y)/ r^2 d\omega
  //     = \int L_E cos(\theta) sin(\theta) d\theta d\phi
  return materialEmission * float(InvPi);
}
Vec3f Scene::getEmissionRadiance(const Vec3f& wo, const RayHit& hit) const
{
  if (hit.interaction.normalGeometric.dot(wo) <= 0.0f) return Vec3f::Zero();
  auto& material = getMaterial(hit);
  Vec3f materialEmission(material.emission[0], material.emission[1], material.emission[2]);
  return materialEmission * float(InvPi);
}
Vec3f Scene::getAlbedo(const SurfaceInteraction& interaction) const
{
  return getBXDF(interaction).getAlbedo(interaction);
}
Vec3f Scene::evalEnvironment(const Ray& ray)
{
  return this->background->evalEmission(ray);
}
bool Scene::hasLight() const
{
  return lightTriangles.size() > 0 || !background->isZero();
}
bool Scene::hasSurfaceLight() const
{
  return lightTriangles.size() > 0;
}
bool Scene::isSurfaceLight(const RayHit& hit) const
{
  if (!hasHitSurface(hit)) return false;
  auto& material = getMaterial(hit);
  Vec3f emission(material.emission[0], material.emission[1], material.emission[2]);
  return emission[0] > 0.0f || emission[1] > 0.0f || emission[2] > 0.0f;
}
bool Scene::isInfiniteAreaLight(const RayHit& hit) const
{
  return !hasHitSurface(hit);
}

std::array<Vec3f, 3> Scene::collectTriangle(std::size_t geomID, std::size_t primID) const
{
  std::array<Vec3f, 3> p;

  const tinyobj::attrib_t& attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

  std::array<std::size_t, 3> idcs = {
    shapes[geomID].mesh.indices[3 * primID + 0].vertex_index,
    shapes[geomID].mesh.indices[3 * primID + 1].vertex_index,
    shapes[geomID].mesh.indices[3 * primID + 2].vertex_index
  };
  for (int v = 0; v < 3; v++)
  {
    tinyobj::real_t vx = attrib.vertices[3 * size_t(idcs[v]) + 0];
    tinyobj::real_t vy = attrib.vertices[3 * size_t(idcs[v]) + 1];
    tinyobj::real_t vz = attrib.vertices[3 * size_t(idcs[v]) + 2];
    p[v] = Vec3f(vx, vy, vz);
  }

  return p;
}
std::array<Vec3f, 3> Scene::collectTriangleNormals(std::size_t geomID, std::size_t primID) const
{
  std::array<Vec3f, 3> vertexNormals;
  const tinyobj::attrib_t& attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

  for (int v = 0; v < 3; v++)
  {
    tinyobj::index_t idx = shapes[geomID].mesh.indices[3 * primID + v];
    if (idx.normal_index >= 0)
    {
      tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
      tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
      tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
      vertexNormals[v] = Vec3f(nx, ny, nz);
    }
    else
    {
      //failed = true;
      std::array<Vec3f, 3> p = collectTriangle(geomID, primID);
      Vec3f n = (p[1] - p[0]).cross(p[2] - p[0]).normalized();
      for (int k = 0; k < 3; k++)
      {
        vertexNormals[k] = n;
      }
      break;
    }
  }
  return vertexNormals;
}
SurfaceInteraction Scene::sampleSurfaceLightPosition(RNG& rng, float* pDensity)
{
  if (lightTriangles.size() == 0)
  {
    std::cout << "error: no light triangle found " << std::endl;
    exit(1);
  }
  (*pDensity) = 1.0f / totalLightArea;

  unsigned int triangleIdx = (*lightDistribution)(rng);
  std::array<std::size_t, 2> ids = lightTriangles[triangleIdx];
  unsigned int geomId = ids[0];
  unsigned int primId = ids[1];

  std::array<Vec3f, 3> p = collectTriangle(ids[0], ids[1]);

  Vec3f normal = (p[1] - p[0]).cross(p[2] - p[0]).normalized();

  Vec3f barycentricCoordinates = sampleTriangleUniformly(rng);
  return getInteraction(geomId, primId, barycentricCoordinates, normal);
}
Ray Scene::sampleLightRay(RNG& rng, Vec3f* flux)
{
  // https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping#eq:sppm-particle-weight
  float pPos, pDirection;
  SurfaceInteraction interaction = sampleSurfaceLightPosition(rng, &pPos);
  Vec3f direction = sampleHemisphereCosImportance(rng, interaction.normalGeometric, &pDirection);
  Vec3f L = getEmissionRadiance(direction, interaction.geomID, interaction.primID);
  (*flux) = L * (std::abs(interaction.normalGeometric.dot(direction)) / (pPos * pDirection));
  assertFinite(*flux);
  return constructRay(interaction, direction, true);
}
Ray Scene::sampleLightRayFromStartPoint(RNG& rng, const SurfaceInteraction& point, float* pDensity, RayHit *rayhit, bool *lightVisible)
{
  assert(hasLight());
  Ray resultray;

  EnvironmentTexture* envTexture = dynamic_cast<EnvironmentTexture*>(background.get());
  // we only use sampling of the background when it's a texture
  float pSurfaceSampling = envTexture == nullptr ? 1.0f : (hasSurfaceLight() ? 0.5f : 0.0f) ;
  float xi = rng.uniform01f();
  bool useSurfaceSampling = xi < pSurfaceSampling;
  // p(omega) = p(omega | useSurfaceSampling) p(useSurfaceSampling) + p(omega | !useSurfaceSampling) p(!useSurfaceSampling)

  if (useSurfaceSampling)
  {
    float pDensityA;
    SurfaceInteraction interaction2 = sampleSurfaceLightPosition(rng, &pDensityA);
    Vec3f diff = interaction2.x - point.x;
    Vec3f direction2 = diff.normalized();
    resultray = constructRayEnd(point, interaction2.x, point.normalGeometric.dot(diff) >= 0.0f); // pass end point so we have less numerical issues
  
    rayHit(resultray, rayhit);
    *lightVisible = hasHitSurface(*rayhit, interaction2.geomID, interaction2.primID);
  
    if (*lightVisible)
    {
      (*pDensity) = evalLightRayFromStartPointDensity(point, resultray, *rayhit);
    }
    // if we have hit the infinite area surface light then this was a numerical error that we ignore here, we set lightVisible to false and the sample gets ignored!
  }
  else
  {
    assert(envTexture != nullptr);
    Vec3f direction = envTexture->sample(rng, pDensity);
    resultray = constructRay(point, direction, point.normalGeometric.dot(direction) >= 0.0f);
    rayHit(resultray, rayhit);
    
    bool surfaceLight = isSurfaceLight(*rayhit);
    bool infiniteAreaLight = isInfiniteAreaLight(*rayhit);

    *lightVisible = surfaceLight || infiniteAreaLight;

    if (*lightVisible)
    {
      (*pDensity) = evalLightRayFromStartPointDensity(point, resultray, *rayhit);
    }
  }

  return resultray;
}
float Scene::evalLightRayFromStartPointDensity(const SurfaceInteraction& point, const Ray& ray2, const RayHit& rayhit2)
{
  bool surfaceLight = isSurfaceLight(rayhit2);
  bool infiniteAreaLight = isInfiniteAreaLight(rayhit2);
  if (!surfaceLight && !infiniteAreaLight) return 0.0f;
  
  EnvironmentTexture* envTexture = dynamic_cast<EnvironmentTexture*>(background.get());

  float pSurfaceSampling = envTexture == nullptr ? 1.0f : (hasSurfaceLight() ? 0.5f : 0.0f) ;

  float resultDensity;
  if (surfaceLight)
  {
    float lightStrategyPdfA = computeSurfaceLightProbabilityDensity(rayhit2);

    Vec3f diff = rayhit2.interaction.x - point.x;
    float diffSq = std::max(1e-5f, (ray2.origin() - rayhit2.interaction.x).normSq());
    Vec3f normal2 = rayhit2.interaction.normalGeometric;
    float lightStrategyPdfOmega = lightStrategyPdfA * diffSq / (1e-7f + std::abs(normal2.dot(ray2.direction())));
    resultDensity = pSurfaceSampling * lightStrategyPdfOmega;
    if(envTexture != nullptr)  resultDensity += (1.0f - pSurfaceSampling) * envTexture->evalSamplingDensity(ray2.direction());
  }
  else
  {
    if (envTexture != nullptr) resultDensity = (1.0f - pSurfaceSampling) * envTexture->evalSamplingDensity(ray2.direction());
    else resultDensity = 0.0f;
  }
  assertFinite(resultDensity);
  return resultDensity;
}
float Scene::computeSurfaceLightProbabilityDensity(const RayHit& hit) const
{
  if (hasHitSurface(hit) && isSurfaceLight(hit))
  {
    return 1.0f / totalLightArea;
  }
  else
  {
    return 0.0f;
  }
}
std::string Scene::getObjectName(const RayHit& rayhit)
{
  return reader.GetShapes()[rayhit.rtc.hit.geomID].name;
}

}
