cmake_minimum_required(VERSION 2.8.3)

project(nntrainer)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11 -pthread")
find_package(PkgConfig REQUIRED)

pkg_check_modules(TFLITE tensorflow-lite)
link_libraries(${TFLITE_LIBRARIES})

set(INIPARSER ${PROJECT_SOURCE_DIR}/external/iniparser/src)

include_directories( ${include_directories}
		     ${CMAKE_CURRENT_SOURCE_DIR}
		     ${TFLITE_INCLUDE_DIRS}
		     ${INIPARSER}
		     )

set(SRCS
	src/neuralnet.cpp
	src/matrix.cpp
	src/layers.cpp
	${INIPARSER}/iniparser.c
	${INIPARSER}/dictionary.c
	)

add_library( ${PROJECT_NAME} SHARED ${SRCS} )      
