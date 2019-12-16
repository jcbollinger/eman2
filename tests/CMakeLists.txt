FILE(GLOB testprograms "${CMAKE_CURRENT_SOURCE_DIR}/*")
INSTALL(PROGRAMS ${testprograms}
  DESTINATION    tests/
  COMPONENT      PythonFiles
)
