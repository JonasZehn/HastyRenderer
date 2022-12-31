#include <Hasty/VMath.h>

#include <nlohmann/json.hpp>

namespace Hasty
{
const double Pi = 3.1415926535897932384;
const double InvPi = 1.0 / 3.1415926535897932384;

void from_json(const nlohmann::json& j, Vec3f& v)
{
  for(int i = 0; i < v.size(); i++)
  {
    v[i] = j.at(i).get<std::remove_reference<decltype(v)>::type::ScalarType>();
  }
}

}