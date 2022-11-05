project('OpenImageDenoise', 'cpp', version : '3.0', default_options : ['warning_level=3', 'cpp_std=c++14'])

cppc = meson.get_compiler('cpp')
OpenImageDenoise_lib = meson.current_source_dir() / 'lib/OpenImageDenoise.lib'
tbb_lib = meson.current_source_dir() / 'lib/tbb.lib'

core_inc = [include_directories('./include')]

OpenImageDenoise_dep = declare_dependency(
	include_directories : core_inc,
	link_args : [tbb_lib, OpenImageDenoise_lib]
)

binpath = meson.current_source_dir() / 'bin'
meson.add_devenv({'PATH': binpath}, method: 'prepend')

