cmake_minimum_required(VERSION 3.14.7)

project(minst LANGUAGES C VERSION 0.1 DESCRIPTION "A library for reading MINST datasets.")

option(MINST_NO_WARNINGS "Whether or not to disable compiler warnings." OFF)
option(MINST_PYTHON      "Whether or not to build the Python bindings." OFF)
option(MINST_DEMO        "Whether or not to build the demo programs." ON)

add_library(minst
  minst.h
  minst.c)

target_include_directories(minst
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

if(CMAKE_COMPILER_IS_GNUCC AND NOT MINST_NO_WARNINGS)
  target_compile_options(minst
    PRIVATE
      -Wall -Werror -Wfatal-errors -Wconversion -ansi -pedantic)
endif()

if(MINST_PYTHON)

  enable_language(CXX)

  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()

  include(FetchContent)

  FetchContent_Declare(pybind11 URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip")
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(pyminst MODULE
    pyminst.cpp)

  targeT_link_libraries(pyminst PUBLIC minst)

endif()

if(MINST_DEMO)
  add_executable(minst_demo demo/c/main.c)
  target_link_libraries(minst_demo PUBLIC minst)
endif()
