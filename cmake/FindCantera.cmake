# - Check for the presence of CANTERA
#
# The following variables are set when CANTERA is found:
#  CANTERA_FOUND       = Set to true, if all components of CANTERA
#                          have been found.
#  CANTERA_INCLUDE_DIR   = Include path for the header files of CANTERA
#  CANTERA_LIBRARIES  = Link these to use CANTERA

## -----------------------------------------------------------------------------
## Check for the cantera include
if(NOT CANTERA_INCLUDE_DIR)
  find_path(CANTERA_INCLUDE_DIR NAMES cantera/Cantera.mak
      HINTS
      ENV CANTERA_ROOT
      ENV CANTERA_ROOT_DIR
      PATHS
      ${CMAKE_INSTALL_PREFIX}/include
      ${KDE4_INCLUDE_DIR}
      PATH_SUFFIXES include
      )
endif(NOT CANTERA_INCLUDE_DIR)

## -----------------------------------------------------------------------------
## Check for the cantera library
if(NOT CANTERA_LIBRARIES)
  find_library(CANTERA_LIBRARIES
      NAMES libcantera.a
      HINTS
      ENV CANTERA_ROOT
      ENV CANTERA_ROOT_DIR
      PATHS
      ${CANTERA_INSTALL_ROOT}
      PATH_SUFFIXES lib
      )
endif(NOT CANTERA_LIBRARIES)


## -----------------------------------------------------------------------------
## Actions taken when all components have been found

if (CANTERA_INCLUDE_DIR AND CANTERA_LIBRARIES)
  set(CANTERA_FOUND TRUE)
else (CANTERA_INCLUDE_DIR AND CANTERA_LIBRARIES)
  if (NOT CANTERA_FIND_QUIETLY)
    if (NOT CANTERA_INCLUDE_DIR)
      message(STATUS "Unable to find CANTERA header files!")
    endif (NOT CANTERA_INCLUDE_DIR)
    if (NOT CANTERA_LIBRARIES)
      message(STATUS "Unable to find CANTERA library files!")
    endif (NOT CANTERA_LIBRARIES)
  endif (NOT CANTERA_FIND_QUIETLY)
endif (CANTERA_INCLUDE_DIR AND CANTERA_LIBRARIES)

if (CANTERA_FOUND)
  if (NOT CANTERA_FIND_QUIETLY)
    message(STATUS "Found components for CANTERA")
    message(STATUS "CANTERA_INCLUDE_DIR = ${CANTERA_INCLUDE_DIR}")
    message(STATUS "CANTERA_LIBRARIES     = ${CANTERA_LIBRARIES}")
  endif (NOT CANTERA_FIND_QUIETLY)
else (CANTERA_FOUND)
  if (CANTERA_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find CANTERA!")
  endif (CANTERA_FIND_REQUIRED)
endif (CANTERA_FOUND)

mark_as_advanced(
    CANTERA_FOUND
    CANTERA_LIBRARIES
    CANTERA_INCLUDE_DIR
)
