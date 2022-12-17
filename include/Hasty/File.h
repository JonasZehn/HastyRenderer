#pragma once

#include <fstream>
#include <vector>
#include <filesystem>

namespace Hasty
{

inline std::vector<char> readFile(const std::filesystem::path& filename) {
  std::ifstream filestream(filename, std::ios::binary);

  if (!filestream.is_open()) {
    throw std::runtime_error("failed to open file '" + filename.string() + "'!");
  }

  filestream.seekg(0, filestream.end);
  size_t fileSize = (size_t)filestream.tellg();
  std::vector<char> buffer(fileSize);

  filestream.seekg(0);
  filestream.read(buffer.data(), fileSize);

  filestream.close();

  return buffer;
}

}
