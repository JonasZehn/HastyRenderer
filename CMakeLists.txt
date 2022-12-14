cmake_minimum_required(VERSION 3.22.0)

include(FetchContent)

project(HastyRenderer VERSION 0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (CMAKE_CXX_STANDARD 17)


option(BUILD_GPU "build gpu path tracer using Vulkan" OFF)

if(BUILD_GPU)
    find_package(Vulkan REQUIRED)
endif()

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


file(GLOB_RECURSE source_files core/include/*.h core/src/*.cpp)

add_library(libHastyRenderer STATIC ${source_files})
target_include_directories(libHastyRenderer PUBLIC core/include)
target_link_libraries(
    libHastyRenderer
    PUBLIC
        nlohmann_json::nlohmann_json
    )
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

if(BUILD_GPU)
    add_executable(VulkanPathTracePrototype "apps/VulkanPathTracePrototype.cpp" "vulkan/src/Vulkan.cpp" "vulkan/include/Hasty/Vulkan.h" "vulkan/include/VulkanInitializers.hpp")
    target_link_libraries(
        VulkanPathTracePrototype
        PUBLIC
            tinyobjloader::tinyobjloader
            libHastyRendererViewer
            Vulkan::Vulkan
        )
    target_include_directories(VulkanPathTracePrototype PRIVATE ${CMAKE_CURRENT_LIST_DIR}/apps ${CMAKE_CURRENT_LIST_DIR}/vulkan/include)
    target_compile_definitions(VulkanPathTracePrototype PRIVATE "HASTY_SHADER_PATH=std::filesystem::path(\"${CMAKE_CURRENT_LIST_DIR}/shaders\")")
endif()
