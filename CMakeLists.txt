cmake_minimum_required(VERSION 2.8.3)

project(nntrainer)
set(NNTRAINER_MAJOR 0)
set(NNTRAINER_MINOR 0)
set(NNTRAINER_RELEASE 1)
set(NNTRAINER_VERSION ${NNTRAINER_MAJOR}.${NNTRAINER_MINOR}.${NNTRAINER_RELEASE})

option(TIZEN "Build for Tizen" OFF)

set (BLASLIB blas-openblas)
if(TIZEN)
  set (BLASLIB openblas)
endif(TIZEN)

if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib/)
endif(NOT DEFINED CMAKE_INSTALL_LIBDIR)

set (PKGCONFIG_INSTALL_DIR "/usr/lib/pkgconfig/" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11 -pthread")

find_package(PkgConfig REQUIRED)
pkg_check_modules(INIPARSER iniparser)
link_libraries(${INIPARSER_LIBRAIRES})
include_directories(${INIPARSER_INCLUDE_DIRS})

option(USE_BLAS "Use BLAS library" ON)

set(SRCS
  src/neuralnet.cpp
  src/tensor.cpp
  src/layers.cpp
  src/databuffer.cpp
  )

set(NNTRAINER_HEADERS
  include/neuralnet.h
  include/tensor.h
  include/layers.h
  include/databuffer.h
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

if(USE_BLAS)
  ADD_DEFINITIONS(-DUSE_BLAS)
  pkg_check_modules(BLAS ${BLASLIB})
  link_libraries(${BLAS_LIBRARIES})
  include_directories(${BLAS_INCLUDE_DIRS})
  message("[-DUSE_BLAS] is enabled")
endif(USE_BLAS)

add_library( ${PROJECT_NAME} SHARED ${SRCS} )

configure_file("package.pc.in" "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc" @ONLY)
install (FILES "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

install (TARGETS nntrainer
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install (FILES ${NNTRAINER_HEADERS} DESTINATION include/nntrainer)
