. cmake_vars.sh

cmake -B $builddir -S . -DCMAKE_TOOLCHAIN_FILE=$vcpkgdir/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=$buildtype
cmake --build $builddir --config $buildtype
