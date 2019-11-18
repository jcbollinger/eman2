CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(EMAN
		VERSION 2.31
		LANGUAGES CXX
		)

include(cmake/functions.cmake)

OPTION(DEBUG_CMAKE_TARGETS "enable debug output for cmake target properties" OFF)
if(DEBUG_CMAKE_TARGETS)
	set(CMAKE_DEBUG_TARGET_PROPERTIES
		INCLUDE_DIRECTORIES
		COMPILE_DEFINITIONS
		COMPILE_OPTIONS
		COMPILE_FEATURES
		LINK_DIRECTORIES
		LINK_OPTIONS
		SOURCES
		POSITION_INDEPENDENT_CODE
		CONTAINER_SIZE_REQUIRED
		LIB_VERSION
		)
endif()

# for 64-bit large file support
ADD_DEFINITIONS(-D_LARGEFILE_SOURCE)
ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64)
ADD_DEFINITIONS(-D_LARGE_FILES)

INCLUDE(${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake)
INCLUDE(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
INCLUDE(${CMAKE_ROOT}/Modules/CheckTypeSize.cmake)

EMAN_CHECK_FUNCTION(fseek64 HAVE_FSEEK64)
EMAN_CHECK_FUNCTION(fseeko HAVE_FSEEKO)
EMAN_CHECK_FUNCTION(ftell64 HAVE_FTELL64)
EMAN_CHECK_FUNCTION(ftello HAVE_FTELLO)
# end for 64-bit large file support

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

MARK_AS_ADVANCED(CLEAR CMAKE_VERBOSE_MAKEFILE)

# Set EMAN_PREFIX
if("$ENV{CONDA_BUILD_STATE}" STREQUAL "BUILD" )
	message("ENV{CONDA_BUILD_STATE}: $ENV{CONDA_BUILD_STATE}")
	if(NOT WIN32)
		set(EMAN_PREFIX $ENV{PREFIX})
	else()
		set(EMAN_PREFIX $ENV{LIBRARY_PREFIX})
	endif()
else()
	find_package(Conda REQUIRED)
		
	if(CONDA_PREFIX)
		if(NOT WIN32)
			set(EMAN_PREFIX ${CONDA_PREFIX})
		else()
			set(EMAN_PREFIX ${CONDA_PREFIX}/Library)
		endif()
	endif()
endif()

set(EMAN_PREFIX_INC      ${EMAN_PREFIX}/include)
set(EMAN_PREFIX_LIB      ${EMAN_PREFIX}/lib)
set(CMAKE_INSTALL_PREFIX ${EMAN_PREFIX} CACHE PATH "installation prefix" FORCE)
set(CMAKE_PREFIX_PATH    ${EMAN_PREFIX} ${EMAN_PREFIX}/..)
cmake_print_variables(CMAKE_PREFIX_PATH)

include(cmake/find_all.cmake)

set(CMAKE_INSTALL_RPATH "${SP_DIR};${EMAN_PREFIX_LIB}")

#flags used for debugging and testing
OPTION(ENABLE_MEMDEBUG "enable memory debug support of objects" OFF)
OPTION(ENABLE_MEMDEBUG2 "enable memory debug support of EMData" OFF)
OPTION(ENABLE_BOUNDS_CHECKING "enable bounds checking support" OFF)
OPTION(ENABLE_RT "enable RT support" ON)
OPTION(ENABLE_WARNINGS "display warnings during compilation" OFF)
OPTION(ENABLE_IOCACHE "enable ImageIO caching" OFF)

#flags for optimization level. You can only turn one of following option to ON, or leave all of them to OFF.
OPTION(ENABLE_DEBUG "enable debug support" OFF)
OPTION(ENABLE_PROFILE "enable symbol generation without debug support" OFF)
OPTION(ENABLE_OPTIMIZE_X86_64 "GCC optimizations for generic X86_64" OFF)
OPTION(ENABLE_OPTIMIZE_MACHINE "Optimize code for this specific computer (typ 5-10% speedup). Binaries may incompatible with some other computers." OFF)
OPTION(ENABLE_OPTIMIZE_PHI "Optimize code for the Intel compiler on the PHI" OFF)
OPTION(ENABLE_OPTIMIZE_WINDOWS_VC "Windows visual c++ specific optimization" OFF)

set(optimization_options
		ENABLE_DEBUG
		ENABLE_PROFILE
		ENABLE_OPTIMIZE_X86_64
		ENABLE_OPTIMIZE_MACHINE
		ENABLE_OPTIMIZE_PHI
		ENABLE_OPTIMIZE_WINDOWS_VC
	)

check_if_options_on_greater_than("${optimization_options}" 1)

IF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
	OPTION(ENABLE_OPTIMIZE_OSX "enable OSX support" ON)
ELSE()
	OPTION(ENABLE_OPTIMIZE_OSX "enable OSX support" OFF)
ENDIF()

if(NOT WIN32)
	set(WARNING_FLAGS_ON  "-W -Wall")
	set(WARNING_FLAGS_OFF "-w")
else()
	set(WARNING_FLAGS_ON  "")    # Default for VS2008 is /W3, highest is /W4
	set(WARNING_FLAGS_OFF "/W0")
endif()

if(ENABLE_WARNINGS)
	add_definitions(${WARNING_FLAGS_ON})
else()
	add_definitions(${WARNING_FLAGS_OFF})
endif()

IF(ENABLE_MEMDEBUG)
	ADD_DEFINITIONS(-DMEMDEBUG)
ENDIF()

IF(ENABLE_MEMDEBUG2)
	ADD_DEFINITIONS(-DMEMDEBUG2)
ENDIF()

IF(ENABLE_V4L2)
	ADD_DEFINITIONS(-DBOUNDS_CHECKING -DENABLE_V4L2)
ENDIF()

IF(ENABLE_BOUNDS_CHECKING)
	ADD_DEFINITIONS(-DBOUNDS_CHECKING)
ENDIF()

IF(ENABLE_DEBUG)
	SET (OPT_FLAGS "-DDEBUG -O2 -g -ansi -fno-inline")
ELSEIF(ENABLE_PROFILE)
	SET (OPT_FLAGS "-O2 -ansi -march=native -m64 -ffast-math -pipe")
	OPTION(ENABLE_DEBUG "enable debug support" OFF)
ELSEIF(ENABLE_OPTIMIZE_OSX)
#	SET (OPT_FLAGS "-O3 -ffast-math")
	SET (OPT_FLAGS "-O3")
#	SET (OPT_FLAGS "-Os -ffast-math")
ELSEIF(ENABLE_OPTIMIZE_MACHINE)
	SET (OPT_FLAGS "-O5 -ansi -march=native -m64 -ffast-math -pipe")
ELSEIF(ENABLE_OPTIMIZE_X86_64)
	SET (OPT_FLAGS "-O5 -ansi -march=k8 -mfpmath=sse -m64 -ffast-math -pipe")
#	SET (OPT_FLAGS "-O5 -ansi -mfpmath=both -march=core-avx-i -m64 -ffast-math -pipe")
ELSEIF(ENABLE_OPTIMIZE_WINDOWS_VC)
	SET (OPT_FLAGS "/Ox /DHAVE_INLINE")
ELSEIF(ENABLE_OPTIMIZE_PHI)
	SET (OPT_FLAGS "-fast")
#	SET (OPT_FLAGS "-O3 -xHost")
ELSE()
	IF(WIN32)
		SET(OPT_FLAGS "")
	ELSE()
		SET (OPT_FLAGS "-O3 -ansi -ffast-math")     # This is the default
	ENDIF()
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
#	SET(EMAN_CXX_FLAGS " " CACHE INTERNAL "EMAN CXX FLAGS")
	IF(ENABLE_DEBUG)
		SET(EMAN_CXX_FLAGS "-g -Wcast-qual -Wwrite-strings" CACHE INTERNAL "EMAN CXX FLAGS")
		SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")
		SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")
	ELSE()
		SET(EMAN_CXX_FLAGS "" CACHE INTERNAL "EMAN CXX FLAGS")
	ENDIF()
	SET(EMAN_CXX_FLAGS "${EMAN_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0" CACHE INTERNAL "EMAN CXX FLAGS")
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMAN_CXX_FLAGS} ${OPT_FLAGS}")

ADD_SUBDIRECTORY(libpyEM)
