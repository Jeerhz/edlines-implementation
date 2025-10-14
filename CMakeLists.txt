cmake_minimum_required(VERSION 3.11)
project(EDStudy)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(EDLib STATIC
        EDLines.cpp
        EDLines.h
        NFA.cpp
        ED-perso.cpp
        ED-perso.h
        PPoint.h
        PPoint.cpp
        Chain.cpp
        Chain.h
        Stack.cpp
        Stack.h
        NFA.cpp
        NFA.h)


add_executable(test_ED test_ED.cpp)
target_link_libraries(test_ED ${OpenCV_LIBS} EDLib)