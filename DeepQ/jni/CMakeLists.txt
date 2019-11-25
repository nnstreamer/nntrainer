cmake_minimum_required(VERSION 2.8.3)
project(DeepQ)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11")

ADD_DEFINITIONS(-DUSING_CUSTOM_ENV)

find_package(PkgConfig REQUIRED)

pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

pkg_check_modules(CURL libcurl)
link_libraries(${CURL_LIBRARIES})

set(ENVDIR ../../Environment)
set(NEURALNET ../../NeuralNet)
set(INIPARSER ../../iniparser/src)

include_directories( ${include_directories}
                     ${CMAKE_CURRENT_SOURCE_DIR}/include
		     ${JSONCPP_INCLUDE_DIRS}
		     ${ENVDIR}
		     ${NEURALNET}
		     ${INIPARSER}
		     )
set(SRCS
	main.cpp
	gym_binding.cpp
	${ENVDIR}/CartPole/cartpole.cpp
	${NEURALNET}/neuralnet.cpp
	${NEURALNET}/matrix.cpp
	${NEURALNET}/layers.cpp
	${INIPARSER}/iniparser.c
	${INIPARSER}/dictionary.c
	)

add_executable(DeepQ ${SRCS})
target_link_libraries( DeepQ ${JSONCPP_LIBRARIES} ${CURL_LIBRARIES})

install (TARGETS DeepQ
	RUNTIME DESTINATION bin
	)
