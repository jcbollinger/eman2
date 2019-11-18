function(ADD_PYSTE_LIB libname)
	add_library(${libname} SHARED lib${libname}.cpp)
	
	if(WIN32)
		set(suffix .pyd)
	else()
		set(suffix .so)
	endif()

	set_target_properties(${libname}
			PROPERTIES
			PREFIX lib
			SUFFIX ${suffix}
			LINK_LIBRARIES "Boost::python"
			)
	
	install(TARGETS ${libname} DESTINATION ${SP_DIR})
endfunction()

if(ENABLE_OPENGL)
	ADD_PYSTE_LIB(pyGLUtils2)
	target_link_libraries(pyGLUtils2       EMAN::OpenGL FTGL::FTGL)
endif()
