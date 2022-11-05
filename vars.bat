rem setting up visual studio variables
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

rem https://mesonbuild.com/Builtin-options.html
set buildtype=debugoptimized
set builddir=build-%buildtype%
rem set backend=ninja
set backend=vs

set ZLIB_INCLUDE_DIR=D:/Code/zlib/1.2.11/build/install/include
set ZLIB_LIBRARY=D:/Code/zlib/1.2.11/build/install/lib/zlibstatic.lib
