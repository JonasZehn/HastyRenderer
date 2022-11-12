#include <Hasty/tinyobj.h>

namespace Hasty
{

// assumes there are only triangles in the input
std::array<Vec3f, 3> collectTriangle(const tinyobj::ObjReader &reader, std::size_t geomID, std::size_t primID)
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
// assumes there are only triangles in the input
std::array<Vec3f, 3> collectTriangleNormals(const tinyobj::ObjReader &reader, std::size_t geomID, std::size_t primID)
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
      std::array<Vec3f, 3> p = collectTriangle(reader, geomID, primID);
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
// assumes there are only triangles in the input
std::array<Vec3f, 3> collectTriangleVec3f(const tinyobj::ObjReader &reader, const std::vector<Vec3f> property_, std::size_t geomID, std::size_t primID)
{
  std::array<Vec3f, 3> property_F;
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

  for (int v = 0; v < 3; v++)
  {
    unsigned int vertexIdx = shapes[geomID].mesh.indices[3 * primID + v].vertex_index;
    property_F[v] = property_[vertexIdx];
  }
  return property_F;
}
std::optional< std::array<Vec2f, 3> > collectTriangleUV(const tinyobj::ObjReader &reader, std::size_t geomID, std::size_t primID)
{
  const tinyobj::attrib_t& attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();
  std::array<Vec2f, 3> uv;
  for (int v = 0; v < 3; v++)
  {
    tinyobj::index_t idx = shapes[geomID].mesh.indices[3 * primID + v];
    if (2 * size_t(idx.texcoord_index) + 1 >= attrib.texcoords.size())
    {
      return std::optional< std::array<Vec2f, 3> >();
    }
    Vec2f uv_j(attrib.texcoords[2 * size_t(idx.texcoord_index) + 0], attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]);
    uv[v] = uv_j;
  }
  return uv;
}


}

