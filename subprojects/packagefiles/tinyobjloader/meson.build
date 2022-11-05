project('tinyobjloader', 'cpp', version : '1.0.7', default_options : ['warning_level=3', 'cpp_std=c++14'])


all_sources = ['./tiny_obj_loader.cc' ]

core_inc = [include_directories('.')]


tinyobjloader_lib = static_library('tinyobjloader',
    all_sources,
    include_directories : core_inc,
    install : true
)


tinyobjloader_dep = declare_dependency(include_directories : core_inc,
  link_with : tinyobjloader_lib)
