set(OpenGL_GL_PREFERENCE GLVND)

find_package(OpenGL REQUIRED COMPONENTS OpenGL)

if(OpenGL_FOUND AND NOT TARGET OpenGL AND NOT TARGET EMAN::OpenGL)
	add_library(OpenGL INTERFACE)
	add_library(EMAN::OpenGL ALIAS OpenGL)

	target_link_libraries(OpenGL INTERFACE OpenGL::GL OpenGL::GLU)
endif()
