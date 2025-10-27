cmake_minimum_required(VERSION 3.11)
cmake_policy(SET CMP0156 NEW)
cmake_policy(SET CMP0181 NEW)

project(EDStudy)

set(CMAKE_CXX_STANDARD 11)
# Enable debugging and AddressSanitizer
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(EDLib STATIC
        EDLines.cpp
        EDLines.h
        NFA.cpp
        ED-perso.cpp
        ED-perso.h
        Chain.cpp
        Chain.h
        Stack.cpp
        Stack.h
        NFA.cpp
        NFA.h
        original-ED/original_ED.cpp
        original-ED/original_ED.h
        original-ED/original_EDLines.cpp
        original-ED/original_EDLines.h
        )


add_executable(test_ED test_ED.cpp)
target_link_libraries(test_ED ${OpenCV_LIBS} EDLib)