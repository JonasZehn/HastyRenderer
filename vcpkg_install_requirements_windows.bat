call cmake_vars.bat

set flags=--triplet %vcpkgtriplet%
%vcpkgdir%/vcpkg install nlohmann-json %flags%
%vcpkgdir%/vcpkg install tinyobjloader %flags%
%vcpkgdir%/vcpkg install embree3 %flags%
%vcpkgdir%/vcpkg install openexr %flags%
%vcpkgdir%/vcpkg install libpng %flags%
%vcpkgdir%/vcpkg install robin-hood-hashing %flags%
%vcpkgdir%/vcpkg install sdl2 %flags%
%vcpkgdir%/vcpkg install benchmark %flags%
%vcpkgdir%/vcpkg install gtest %flags%
