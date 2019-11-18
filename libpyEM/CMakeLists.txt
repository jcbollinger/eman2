function(ADD_PYSTE_LIB libname)
	add_library(${libname} SHARED lib${libname}.cpp)

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
