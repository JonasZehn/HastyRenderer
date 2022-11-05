project('embree', 'cpp', version : '3.0', default_options : ['warning_level=3', 'cpp_std=c++14'])

cppc = meson.get_compiler('cpp')
embree_lib = meson.current_source_dir() / 'lib/embree3.lib'
tbb_lib = meson.current_source_dir() / 'lib/tbb.lib'

core_inc = [include_directories('./include')]

embree_dep = declare_dependency(
	include_directories : core_inc,
	link_args : [tbb_lib, embree_lib]
)

binpath = meson.current_source_dir() / 'bin'
meson.add_devenv({'PATH': binpath}, method: 'prepend')

