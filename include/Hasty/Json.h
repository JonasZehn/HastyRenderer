#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>

namespace Hasty
{

nlohmann::json readJSON(std::filesystem::path jsonFilepath);

template<typename T>
void json_get_optional(T& target, const nlohmann::json& j, const char* s, T defaultVal)
{
  if (j.contains(s))
  {
    target = j.at(s).get<T>();
  }
  else
  {
    target = defaultVal;
  }
}

}