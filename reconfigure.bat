call vars.bat

meson setup %builddir% --backend %backend% --buildtype=%buildtype%  -DZLIB_INCLUDE_DIR=%ZLIB_INCLUDE_DIR% -DZLIB_LIBRARY=%ZLIB_LIBRARY% --reconfigure
