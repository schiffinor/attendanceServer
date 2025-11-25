# Boost (Beast + System + Regex) via CMake Modules

# OpenBLAS (yours)
set(OPENBLAS_ROOT   "C:/numerics/openblas_install")
find_path(OPENBLAS_INCLUDE_DIR cblas.h PATHS "${OPENBLAS_ROOT}/include")
find_library(OPENBLAS_LIB       openblas   PATHS "${OPENBLAS_ROOT}/lib")
if (NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found in ${OPENBLAS_ROOT}")
endif()
include_directories(${OPENBLAS_INCLUDE_DIR})

# Boost (Beast + System + Regex) via CMake Modules
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
# set(Boost_DEBUG 1)
set(BOOST_ROOT   "C:/Program Files/boost")
set(BOOST_INCLUDEDIR   "C:/Program Files/boost")
set(BOOST_LIBRARYDIR   "C:/Program Files/boost/stage/lib")
list(APPEND CMAKE_IGNORE_PATH "C:/Program Files (x86)" "C:/Program Files/LLVM")
set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_COMPILER "-mgw8")
set(BOOST_DETECTED_TOOLSET "-mgw8")
set(Boost_USE_STATIC_LIBS ON)
set(Boost_ARCHITECTURE "-x64")
find_package(Boost REQUIRED COMPONENTS system regex)
if (NOT Boost_FOUND)
    message(FATAL_ERROR "Could not find Boost.System+Regex")
endif()
# message(STATUS "Boost headers: ${Boost_INCLUDE_DIRS}")
# message(STATUS "Boost libs:    ${Boost_LIBRARIES}")

include_directories(${Boost_INCLUDE_DIRS})

# Windows needs Winsock
if (WIN32)
    set(PLATFORM_LIBS ws2_32)
endif()



