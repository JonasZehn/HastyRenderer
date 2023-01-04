#include <Hasty/Json.h>

#include <fstream>

namespace Hasty
{

nlohmann::json readJSON(std::filesystem::path jsonFilepath)
{
  std::ifstream ifs;
  ifs.open(jsonFilepath);
  if(!ifs.is_open())
  {
    throw std::runtime_error(std::string("could not open ") + jsonFilepath.string() + "  for reading ");
  }

  nlohmann::json jsonData;
  ifs >> jsonData;
  return jsonData;
}

}