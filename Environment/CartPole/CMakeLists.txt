cmake_minimum_required(VERSION 2.8.3)
project(CartPole)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11")

include_directories( ${include_directories}
                     ${CMAKE_CURRENT_SOURCE_DIR}
		     )
set(SRCS
	main.cpp
	cartpole.cpp
	)

add_executable(CartPole ${SRCS})

install (TARGETS CartPole
	RUNTIME DESTINATION bin
	)
