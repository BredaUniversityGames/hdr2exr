# Use a modern CMake version
cmake_minimum_required(VERSION 3.24)

# Define the project
project(hdr2exr)

# Set the C++ standard to a modern version
set(CMAKE_CXX_STANDARD 20)

# Add tinyexr as a subdirectory
add_subdirectory(external/tinyexr)

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

# Add the stb include directory, the tinyexr include directory and the Vulkan include directory
include_directories(
    ${CMAKE_SOURCE_DIR}/external/stb
    ${CMAKE_SOURCE_DIR}/external/tinyexr
    ${Vulkan_INCLUDE_DIRS}
)

# Add the executable
add_executable(hdr2exr main.cpp converter.cpp)

# Link the tinyexr library and the Vulkan library
target_link_libraries(hdr2exr
    tinyexr
    ${Vulkan_LIBRARIES})