set(OpenGL_GL_PREFERENCE GLVND)

find_package(OpenGL REQUIRED COMPONENTS OpenGL)

cmake_print_variables(OPENGL_INCLUDE_DIR)
cmake_print_variables(OPENGL_LIBRARIES)
cmake_print_variables(OPENGL_glu_LIBRARY)
cmake_print_variables(OPENGL_opengl_LIBRARY)
cmake_print_variables(OPENGL_gl_LIBRARY)
cmake_print_variables(OPENGL_FOUND)
cmake_print_variables(OPENGL_GLU_FOUND)
cmake_print_variables(OpenGL_OpenGL_FOUND)
cmake_print_variables(OPENGL_opengl_LIBRARY)
cmake_print_variables(OPENGL_gl_LIBRARY)

if(OpenGL_FOUND AND NOT TARGET OpenGL AND NOT TARGET EMAN::OpenGL)
	add_library(OpenGL INTERFACE)
	add_library(EMAN::OpenGL ALIAS OpenGL)

	target_link_libraries(OpenGL INTERFACE OpenGL::GL OpenGL::GLU)
endif()
