call build.bat

if %errorlevel% == 0 meson devenv -C %builddir% %builddir%/PhotonMappingViewer.exe

pause
