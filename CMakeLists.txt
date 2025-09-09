cmake_minimum_required(VERSION 3.30)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

# 1) single project() with CXX enabled
project(attendanceServer
        VERSION 1.0
        LANGUAGES CXX
)

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

add_executable(attendanceServer
        # ${ATTENDANCE_SOURCES}
        main.cpp
)

# tiny demo target
add_executable(testUniqueErase testUniqueErase.cpp)

# zigzag demo target
add_executable(testZigZag testZigZag.cpp)

# zigzag speed test target
add_executable(testZigZagSpeed testZigZagSpeed.cpp)

# server demo target
add_executable(server server.cpp)

# poly demo target
add_executable(polyTest crypto/polyTest.cpp)

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

target_compile_options(
        polyTest
        PRIVATE
        -O3 -mavx2 -mfma
        #-O3 -msse4.2 -mno-avx -mno-avx
)

target_compile_options(
        testZigZag
        PRIVATE
        -O3 -mavx2 -mfma
)

target_compile_options(
        testZigZagSpeed
        PRIVATE
        -O3 -mavx2 -mfma
)

target_link_libraries(attendanceServer
        PRIVATE
        ${Boost_LIBRARIES}
        ${OpenBLAS_LIBRARIES}
        ${PLATFORM_LIBS}
)
