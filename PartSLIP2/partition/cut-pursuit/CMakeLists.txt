# top-level CMake configuration file

cmake_minimum_required(VERSION 3.5)

project(CUT_PURSUIT_SEG)


#------------------------------------------------------------------------------
# internal cmake settings
#------------------------------------------------------------------------------
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_MACOSX_RPATH ON)
include(FeatureSummary)

#------------------------------------------------------------------------------
# General settings
#------------------------------------------------------------------------------
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")

#------------------------------------------------------------------------------
# actual library
#------------------------------------------------------------------------------
add_subdirectory(src)