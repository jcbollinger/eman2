CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(EMAN
		VERSION 2.31
		LANGUAGES CXX
		)

include(CMakePrintHelpers)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Set EMAN_PREFIX
if("$ENV{CONDA_BUILD_STATE}" STREQUAL "BUILD" )
	message("ENV{CONDA_BUILD_STATE}: $ENV{CONDA_BUILD_STATE}")
		set(EMAN_PREFIX $ENV{PREFIX})
endif()

set(CMAKE_PREFIX_PATH    ${EMAN_PREFIX} ${EMAN_PREFIX}/..)

# Find Boost
include(${CMAKE_SOURCE_DIR}/cmake/Boost.cmake)

add_definitions("-w")

function(ADD_PYSTE_LIB libname)
	add_library(${libname} SHARED hello-boost.cpp)

	set_target_properties(${libname}
			PROPERTIES
			PREFIX lib
			SUFFIX .so
			LINK_LIBRARIES "Boost::python"
			)
endfunction()

include(${CMAKE_SOURCE_DIR}/cmake/OpenGL.cmake)

ADD_PYSTE_LIB(pyGLUtils2)
target_link_libraries(pyGLUtils2       EMAN::OpenGL)
