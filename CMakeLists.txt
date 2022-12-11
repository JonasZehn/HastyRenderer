cmake_minimum_required(VERSION 3.22.0)

include(FetchContent)

project(HastyRenderer VERSION 0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(embree 3 CONFIG REQUIRED)
find_package(robin_hood CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)

include(AddShishua)
include(DownloadOpenImageDenoise)
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
        robin_hood::robin_hood
        libshishua
        OpenImageIO::OpenImageIO
        )

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

add_executable(VulkanPrototype "apps/VulkanPrototype.cpp")
target_link_libraries(VulkanPrototype PUBLIC libHastyRendererViewer Vulkan::Vulkan)
target_include_directories(VulkanPrototype PRIVATE ${VULKAN_HPP_INCLUDE_DIRS})
target_include_directories(VulkanPrototype PRIVATE ${CMAKE_CURRENT_LIST_DIR}/apps)
