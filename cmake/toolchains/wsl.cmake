# wsl.cmake — toolchain for building under WSL (x86_64)

# 1) Tell CMake we’re on Linux
set(CMAKE_SYSTEM_NAME    Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)


# 3) Use native make (or ninja if you change generator)
# CMake will pick /usr/bin/make by default for "Unix Makefiles"

# 4) Limit where CMake looks so it stays in WSL’s /usr /usr/local /opt
#    Programs run on build machine, libs/includes/packages only in root paths
set(CMAKE_FIND_ROOT_PATH        /usr /usr/local /opt)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 5) (Optional) If you’ve installed OpenBLAS under /opt/OpenBLAS,
#    you can hint its root here so find_package(OpenBLAS) picks it up:
# ——— OpenBLAS (your existing) ———
set(OPENBLAS_ROOT   "/opt/OpenBLAS")
find_path(OPENBLAS_INCLUDE_DIR cblas.h
        PATHS "${OPENBLAS_ROOT}/include")
find_library(OPENBLAS_LIB openblas
        PATHS "${OPENBLAS_ROOT}/lib")
if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found in ${OPENBLAS_ROOT}")
endif()
include_directories(${OPENBLAS_INCLUDE_DIR})




