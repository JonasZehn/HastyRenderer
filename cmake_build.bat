call cmake_vars.bat

cmake -B %builddir% -S . -DCMAKE_TOOLCHAIN_FILE=%vcpkgdir%/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=%buildtype% -DBUILD_GPU=%buildgpu%
cmake --build %builddir% --config %buildtype%
