cmake_minimum_required(VERSION 2.8.3)

project(nntrainer)

set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include/)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib/)

set (PKGCONFIG_INSTALL_DIR "/usr/lib/pkgconfig/" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11 -pthread")


find_package(PkgConfig REQUIRED)

set(INIPARSER ${PROJECT_SOURCE_DIR}/external/iniparser/src)

include_directories( ${include_directories}
		     ${CMAKE_CURRENT_SOURCE_DIR}
		     ${INIPARSER}
		     )

set(SRCS
	src/neuralnet.cpp
	src/matrix.cpp
	src/layers.cpp
	${INIPARSER}/iniparser.c
	${INIPARSER}/dictionary.c
	)

set(NNTRAINER_HEADERS include/neuralnet.h include/matrix.h include/layers.h)      

add_library( ${PROJECT_NAME} SHARED ${SRCS} )


configure_file("package.pc.in" "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc" @ONLY)
install (FILES "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc" DESTINATION "${PKGCONFIG_INSTALL_DIR}")

install (TARGETS nntrainer
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install (FILES ${NNTRAINER_HEADERS} DESTINATION include)
