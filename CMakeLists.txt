cmake_minimum_required(VERSION 3.30)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

# 1) single project() with CXX enabled
project(attendanceServer
        VERSION 1.0
        LANGUAGES CXX
)

# platform-specific settings for simd
set(SIMD "auto" CACHE STRING "SIMD level: auto|scalar|sse42|avx2")
set_property(CACHE SIMD PROPERTY STRINGS auto scalar sse42 avx2)

function(apply_simd target)
    if (MSVC)
        if (SIMD STREQUAL "avx2")
            target_compile_options(${target} PRIVATE /arch:AVX2)
            target_compile_definitions(${target} PRIVATE CPU_AVX2=1)
        elseif (SIMD STREQUAL "sse42")
            # MSVC doesn't expose /arch:SSE4.2; use SSE2 baseline + your own feature guards
            target_compile_options(${target} PRIVATE /arch:SSE2)
            target_compile_definitions(${target} PRIVATE CPU_SSE42=1)
        elseif (SIMD STREQUAL "scalar")
            target_compile_definitions(${target} PRIVATE CPU_SCALAR=1)
        endif()
    else()
        if (SIMD STREQUAL "avx2")
            target_compile_options(${target} PRIVATE -O3 -mavx2 -mfma)
            target_compile_definitions(${target} PRIVATE CPU_AVX2=1)
        elseif (SIMD STREQUAL "sse42")
            target_compile_options(${target} PRIVATE -O3 -msse4.2 -mno-avx)
            target_compile_definitions(${target} PRIVATE CPU_SSE42=1)
        elseif (SIMD STREQUAL "scalar")
            target_compile_definitions(${target} PRIVATE CPU_SCALAR=1)
        endif()
    endif()
endfunction()

# Create a global definition SIMD_BUILD_CHOICE
add_compile_definitions(SIMD_BUILD_CHOICE="${SIMD}")


# 2) C++ settings
set(CMAKE_CXX_STANDARD       20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS     OFF)

# static libgcc, libstdc++
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
endif()
if (WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "-static")
endif()

# Discourage in-source builds
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(WARNING "In-source builds discouraged; please use a separate build directory.")
endif()


# 3) pull in Boost & OpenBLAS
#    we’ll tell CMake whether to use static Boost via -DBoost_USE_STATIC_LIBS=ON

find_package(Boost 1.88.0 REQUIRED COMPONENTS regex system)
find_package(OpenBLAS REQUIRED)



# 1. glob all .cpp under the source directory
file(GLOB_RECURSE
        ATTENDANCE_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# 2. remove any file whose path contains the build dir
list(FILTER ATTENDANCE_SOURCES EXCLUDE REGEX "cmake-build-")

function(add_utf8_manifest tgt)
    if (WIN32)
        target_sources(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/cmake/utf8.rc)
    endif()
endfunction()

add_executable(attendanceServer
        # ${ATTENDANCE_SOURCES}
        main.cpp
)
add_utf8_manifest(attendanceServer)

# add test executables
# unique erase test demo target
add_executable(testUniqueErase testUniqueErase.cpp)
add_utf8_manifest(testUniqueErase)

# zigzag test demo target
add_executable(testZigZag testZigZag.cpp)
add_utf8_manifest(testZigZag)

# zigzag speed test demo target
add_executable(testZigZagSpeed testZigZagSpeed.cpp)
add_utf8_manifest(testZigZagSpeed)

# server target
add_executable(server server.cpp)
add_utf8_manifest(server)

# poly test target
add_executable(polyTest crypto/polyTest.cpp)
add_utf8_manifest(polyTest)

# 4. exclude test executables from "all"
set_target_properties(testUniqueErase PROPERTIES EXCLUDE_FROM_ALL YES)

target_link_libraries(testUniqueErase PRIVATE)

set_target_properties(testZigZag PROPERTIES EXCLUDE_FROM_ALL YES)

target_link_libraries(testZigZag PRIVATE)

set_target_properties(testZigZagSpeed PROPERTIES EXCLUDE_FROM_ALL YES)

target_link_libraries(testZigZagSpeed PRIVATE)

set_target_properties(server PROPERTIES EXCLUDE_FROM_ALL YES)
# link server to Boost
target_link_libraries(server
        PRIVATE
        ${Boost_LIBRARIES}
)
set_target_properties(polyTest PROPERTIES EXCLUDE_FROM_ALL YES)

# Target compile options and definitions for simd

# Target polyTest
apply_simd(polyTest)

# Target zigzag tests
apply_simd(testZigZag)
apply_simd(testZigZagSpeed)




target_link_libraries(attendanceServer
        PRIVATE
        ${Boost_LIBRARIES}
        ${OpenBLAS_LIBRARIES}
        ${PLATFORM_LIBS}
)
