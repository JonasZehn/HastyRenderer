
FetchContent_Declare(shishua
  GIT_REPOSITORY    https://github.com/espadrine/shishua.git
  GIT_TAG           ed0d60a8b35fb6aba711dcbc24aaeda78c623b45
)
FetchContent_MakeAvailable(shishua)
add_library(libshishua INTERFACE)
target_include_directories(libshishua INTERFACE ${shishua_SOURCE_DIR})
