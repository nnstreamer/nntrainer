cmake_minimum_required(VERSION 2.8.3)
project(LogisticRegression)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11 -pthread")

set(NEURALNET ../../NeuralNet)
set(INIPARSER ../../iniparser/src)

include_directories( ${include_directories}
		     ${CMAKE_CURRENT_SOURCE_DIR}
		     ${NEURALNET}
		     ${INIPARSER}
		     )
		   
set(SRCS
	main.cpp
	${NEURALNET}/neuralnet.cpp
	${NEURALNET}/matrix.cpp
	${NEURALNET}/layers.cpp
	${INIPARSER}/iniparser.c
	${INIPARSER}/dictionary.c
	)

add_executable(LogisticRegression ${SRCS})
target_link_libraries( LogisticRegression ${CMAKE_DL_LIBS})

install (TARGETS LogisticRegression
	RUNTIME DESTINATION bin
	)
