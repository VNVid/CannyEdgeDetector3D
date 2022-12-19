cmake_minimum_required(VERSION 3.1)

project(EdgeDetection3D)

find_package(OpenCV REQUIRED)


#add_library(Canny3D blur.cpp blur.h)
#target_link_libraries(Canny3D PRIVATE ${OpenCV_LIBS})
#add_executable(EdgeDetector main.cpp)
#target_link_libraries(EdgeDetector Canny3D)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(EdgeDetector main.cpp blur.cpp blur.h sobel.cpp sobel.h canny.cpp canny.h) 

target_link_libraries(EdgeDetector PRIVATE ${OpenCV_LIBS})