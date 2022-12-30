cmake_minimum_required(VERSION 3.1)

project(Main)

find_package(OpenCV REQUIRED)

add_subdirectory(lib)
add_executable(main main.cpp)

include_directories(${PROJECT_SOURCE_DIR})

target_link_libraries(main EdgeDetector ${OpenCV_LIBS})
