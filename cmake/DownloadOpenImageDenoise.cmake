if (WIN32)
    FetchContent_Declare(oidn
      URL      https://github.com/OpenImageDenoise/oidn/releases/download/v1.4.3/oidn-1.4.3.x64.vc14.windows.zip
      URL_MD5  649626f3043c6bea0319eecc44676461
    )
elseif(LINUX)
    FetchContent_Declare(oidn
      URL      https://github.com/OpenImageDenoise/oidn/releases/download/v1.4.3/oidn-1.4.3.x86_64.linux.tar.gz
      URL_MD5  B6B255D513C953665DB6D76EC583A13B
    )
elseif(APPLE)
    FetchContent_Declare(oidn
      URL      https://github.com/OpenImageDenoise/oidn/releases/download/v1.4.3/oidn-1.4.3.x86_64.macos.tar.gz
      URL_MD5  6E0B99237E4D97DB42155B91FEE84CEE
    )
endif ()

FetchContent_MakeAvailable(oidn)
list(APPEND CMAKE_PREFIX_PATH "${oidn_SOURCE_DIR}/lib/cmake")
