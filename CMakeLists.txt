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

# Discourage in-source builds
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(WARNING "In-source builds discouraged; please use a separate build directory.")
endif()


# 3) pull in Boost & OpenBLAS
#    we’ll tell CMake whether to use static Boost via -DBoost_USE_STATIC_LIBS=ON

find_package(Boost 1.88.0 REQUIRED COMPONENTS regex system)
find_package(OpenBLAS REQUIRED)





# 4) your executable
add_executable(attendanceServer
        main.cpp
        # …other sources…
)


target_link_libraries(attendanceServer
        PRIVATE
        ${Boost_LIBRARIES}
        ${OpenBLAS_LIBRARIES}
        ${PLATFORM_LIBS}
)
