cmake_minimum_required(VERSION 3.1)

project(EdgeDetection3D)


find_package(OpenCV REQUIRED)

add_executable(eval evaluation.cpp)

target_link_libraries(eval PRIVATE ${OpenCV_LIBS})