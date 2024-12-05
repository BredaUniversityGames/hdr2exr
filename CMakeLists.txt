# Use a modern CMake version
cmake_minimum_required(VERSION 3.20)

# Define the project
project(hdr2exr)

# Set the C++ standard to a modern version
set(CMAKE_CXX_STANDARD 20)

# Add the stb include directory
include_directories(${CMAKE_SOURCE_DIR}/external/stb)

# Add the executable
add_executable(hdr2exr main.cpp converter.cpp)