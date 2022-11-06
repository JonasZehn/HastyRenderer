call meson_vars.bat

if %errorlevel% == 0 meson devenv -C %builddir% %builddir%/PhotonMappingViewer.exe ../renderJob.json .

pause
