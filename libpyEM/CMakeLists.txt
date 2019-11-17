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
			LINK_LIBRARIES "EM2;Boost::python"
			)
	
	install(TARGETS ${libname} DESTINATION ${SP_DIR})
endfunction()

if(ENABLE_OPENGL)
	ADD_PYSTE_LIB(pyGLUtils2)
	ADD_PYSTE_LIB(pyMarchingCubes2)
	
	target_sources(pyMarchingCubes2 PRIVATE
				   ${CMAKE_SOURCE_DIR}/libEM/marchingcubes.cpp
				   )
	target_sources(pyGLUtils2 PRIVATE
				   ${CMAKE_SOURCE_DIR}/libEM/emftgl.cpp
				   ${CMAKE_SOURCE_DIR}/libEM/glutil.cpp
				   ${CMAKE_SOURCE_DIR}/libEM/marchingcubes.cpp
				   )
	target_link_libraries(pyGLUtils2       EMAN::OpenGL FTGL::FTGL)
	target_link_libraries(pyMarchingCubes2 EMAN::OpenGL)
endif()

file(GLOB empythonlibs "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(FILES ${empythonlibs}
		DESTINATION ${SP_DIR}
		COMPONENT PythonFiles
		)

add_subdirectory(qtgui)
