cmake_minimum_required(VERSION 3.1)

project(EdgeDetection3D)

find_package(OpenCV REQUIRED)


set(SOURCES blur.cpp sobel.cpp canny.cpp )
set(HEADERS blur.h  sobel.h canny.h)
add_library(EdgeDetector ${SOURCES} ${HEADERS})

include_directories(${PROJECT_SOURCE_DIR})

target_link_libraries(EdgeDetector PRIVATE ${OpenCV_LIBS})

target_include_directories(EdgeDetector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})