#pragma once

#include <Hasty/VMath.h>

#include <tiny_obj_loader.h>

#include <array>
#include <optional>

namespace Hasty
{

std::array<Vec3f, 3> collectTriangle(const tinyobj::ObjReader& reader, std::size_t geomID, std::size_t primID);
std::array<Vec3f, 3> collectTriangleNormals(const tinyobj::ObjReader& reader, std::size_t geomID, std::size_t primID);
std::array<Vec3f, 3> collectTriangleVec3f(const tinyobj::ObjReader& reader, const std::vector<Vec3f> property_, std::size_t geomID, std::size_t primID);
std::optional< std::array<Vec2f, 3> > collectTriangleUV(const tinyobj::ObjReader& reader, std::size_t geomID, std::size_t primID);

}
