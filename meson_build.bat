call meson_vars.bat

REM if not exist  "builddir\" meson setup %builddir% --backend %backend% --buildtype=%buildtype%
meson setup %builddir% --backend %backend% --buildtype=%buildtype% -DZLIB_INCLUDE_DIR=%ZLIB_INCLUDE_DIR% -DZLIB_LIBRARY=%ZLIB_LIBRARY%

meson compile -C %builddir%
