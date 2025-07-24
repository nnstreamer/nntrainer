cmake_minimum_required(VERSION 2.8.3)
project(LogisticRegression)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11 -pthread")

find_package(PkgConfig REQUIRED)

pkg_check_modules(INIPARSER iniparser)
link_libraries(${INIPARSER_LIBRARIES})

pkg_check_modules(NNTRAINER nntrainer)
link_libraries(${NNTRAINER_LIBRARIES})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${NNTRAINER_INCLUDE_DIRS}
  ${INIPARSER_INCLUDE_DIRS}
  )
		   
set(SRCS
  main.cpp
  )

add_executable(LogisticRegression ${SRCS})
target_link_libraries( LogisticRegression ${CMAKE_DL_LIBS} ${NNTRAINER_LIBRARIES} ${INIPARSER_LIBRARIES})

install (TARGETS LogisticRegression
	RUNTIME DESTINATION bin
	)
