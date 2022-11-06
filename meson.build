project('HastyRenderer', 'cpp', version : '0.1', default_options : ['warning_level=3', 'cpp_std=c++17'])

cppc = meson.get_compiler('cpp')

# openexr requires zlib
cmake = import('cmake')
opt_var = cmake.subproject_options()
opt_var.add_cmake_defines({'ZLIB_INCLUDE_DIR': get_option('ZLIB_INCLUDE_DIR')})
opt_var.add_cmake_defines({'ZLIB_LIBRARY': get_option('ZLIB_LIBRARY')})

openexr_proj = cmake.subproject('openexr', options: opt_var)
openexr_dep = openexr_proj.dependency('OpenEXR')

embree_dep = dependency('embree')
OpenImageDenoise_dep = dependency('OpenImageDenoise')
tinyobjloader_dep = dependency('tinyobjloader')
sdl2_dep = dependency('sdl2')
libpng_dep = dependency('libpng')
shishua_dep = dependency('shishua')
nlohmann_json_dep = dependency('nlohmann_json')
gtest_dep = dependency('gtest')
robin_hood_dep = dependency('robin_hood')
benchmark_dep = dependency('benchmark')

incdir = include_directories('include')
header_folder = 'include/Hasty'
lib_src_files = [
	header_folder / 'BRDF.h', 'src/BRDF.cpp',
	header_folder / 'Camera.h', 'src/Camera.cpp',
	header_folder / 'Environment.h', 'src/Environment.cpp',
	header_folder / 'HashCells.h',
	header_folder / 'Image.h', 'src/Image.cpp',
	header_folder / 'Json.h', 'src/Json.cpp',
	header_folder / 'LightRayInfo.h', 'src/LightRayInfo.cpp',
	header_folder / 'MIS.h', 'src/MIS.cpp',
	header_folder / 'PathTracing.h', 'src/PathTracing.cpp',
	header_folder / 'PhotonMapping.h', 'src/PhotonMapping.cpp',
	header_folder / 'Random.h', 'src/Random.cpp',
	header_folder / 'RenderJob.h', 'src/RenderJob.cpp',
	header_folder / 'Sample.h', 'src/Sample.cpp',
	header_folder / 'Scene.h', 'src/Scene.cpp',
	header_folder / 'SmallVector.h',
	header_folder / 'Timer.h',
	header_folder / 'Texture.h', 'src/Texture.cpp',
	header_folder / 'VMath.h', 'src/VMath.cpp']
hr_lib = static_library('HastyRenderer',
					lib_src_files,
					include_directories : incdir,
					dependencies: [embree_dep, tinyobjloader_dep, shishua_dep, nlohmann_json_dep, robin_hood_dep, openexr_dep, libpng_dep]
					)

hr_dep = declare_dependency(
    link_with : hr_lib,
    include_directories : incdir,
	dependencies: [embree_dep, tinyobjloader_dep, shishua_dep, nlohmann_json_dep, robin_hood_dep, openexr_dep, libpng_dep]
)


lib_src_files = [
	'apps/ViewerUtils.cpp', 'apps/ViewerUtils.h']
incdir = include_directories('apps')
hr_viewer_lib = static_library('HastyRendererViewer',
					lib_src_files,
					include_directories : incdir,
					dependencies: [sdl2_dep, OpenImageDenoise_dep, hr_dep]
					)
hr_viewer_dep = declare_dependency(
    link_with : hr_viewer_lib,
    include_directories : incdir,
	dependencies: [sdl2_dep, openexr_dep, OpenImageDenoise_dep, hr_dep]
)

app_srcfiles = ['apps/PathTracingViewer.cpp']
executable('PathTracingViewer',
			app_srcfiles,
			dependencies: [hr_viewer_dep]
			)
app_srcfiles = ['apps/PhotonMappingViewer.cpp' ]
executable('PhotonMappingViewer',
			app_srcfiles,
			dependencies: [hr_viewer_dep]
			)

test_executable = executable('HastyTests',
			'apps/Tests.cpp',
			dependencies: [hr_dep, gtest_dep])
test('Tests', test_executable)

benchmarks_executable = executable('HastyBenchmarks',
			'apps/Benchmarks.cpp',
			dependencies: [hr_dep,benchmark_dep])
