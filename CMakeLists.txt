cmake_minimum_required(VERSION 3.24.0)

include(FetchContent)

project(HastyRenderer VERSION 0.1 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(embree 3 CONFIG REQUIRED)
find_package(OpenEXR CONFIG REQUIRED)
find_package(PNG REQUIRED)
find_package(robin_hood CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
 
FetchContent_Declare(shishua
  GIT_REPOSITORY    https://github.com/espadrine/shishua.git
  GIT_TAG           ed0d60a8b35fb6aba711dcbc24aaeda78c623b45
)
FetchContent_MakeAvailable(shishua)
add_library(libshishua INTERFACE)
target_include_directories(libshishua INTERFACE ${shishua_SOURCE_DIR})
 
FetchContent_Declare(oidn
  URL      https://github.com/OpenImageDenoise/oidn/releases/download/v1.4.3/oidn-1.4.3.x64.vc14.windows.zip
  URL_MD5  649626f3043c6bea0319eecc44676461
)
FetchContent_MakeAvailable(oidn)
list(APPEND CMAKE_PREFIX_PATH "${oidn_SOURCE_DIR}/lib/cmake")
find_package(OpenImageDenoise CONFIG REQUIRED)

 
file(GLOB_RECURSE source_files include/*.h src/*.cpp)

add_library(libHastyRenderer STATIC ${source_files})
target_include_directories(libHastyRenderer PUBLIC include)
target_link_libraries(libHastyRenderer PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(
    libHastyRenderer
    PRIVATE
        tinyobjloader::tinyobjloader
        embree
        OpenEXR::OpenEXR
        PNG::PNG
        robin_hood::robin_hood
        libshishua)

set(source_files "apps/ViewerUtils.cpp" "apps/ViewerUtils.h")
add_library(libHastyRendererViewer STATIC ${source_files})
target_include_directories(libHastyRendererViewer PUBLIC apps)
target_link_libraries(
    libHastyRendererViewer
    PUBLIC
        libHastyRenderer
        OpenImageDenoise
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>)

set(source_files "apps/PathTracingViewer.cpp")
add_executable(PathTracingViewer ${source_files})
target_link_libraries(PathTracingViewer PUBLIC libHastyRendererViewer)

set(source_files "apps/PhotonMappingViewer.cpp")
add_executable(PhotonMappingViewer ${source_files})
target_link_libraries(PhotonMappingViewer PUBLIC libHastyRendererViewer)

set(source_files "apps/Tests.cpp")
add_executable(HastyTests ${source_files})
target_link_libraries(HastyTests PUBLIC libHastyRendererViewer GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main)

set(source_files "apps/Benchmarks.cpp")
add_executable(HastyBenchmarks ${source_files})
target_link_libraries(
    HastyBenchmarks
    PUBLIC
        libHastyRendererViewer
        benchmark::benchmark
        benchmark::benchmark_main)
target_compile_definitions(HastyBenchmarks PUBLIC BENCHMARK_STATIC_DEFINE=1)

