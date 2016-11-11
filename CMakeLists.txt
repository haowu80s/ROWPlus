cmake_minimum_required(VERSION 3.6)
project(ROWPlus)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#
#set(SOURCE_FILES main.cpp)
#add_executable(ROWPlus ${SOURCE_FILES})

# guard against in-source builds

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# guard against bad build-type strings

#if (NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE "Release")
#endif()
#string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
#if(    NOT cmake_build_type_tolower STREQUAL "debug"
#    AND NOT cmake_build_type_tolower STREQUAL "release"
#    AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
#  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
#endif()


#############################################################################
# retrieve version infomation                                               #
#############################################################################

include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#############################################################################
# find how to link to the standard libraries                                #
#############################################################################

find_package(StandardMathLibrary)

set(ROWPLUS_TEST_CUSTOM_LINKER_FLAGS  "" CACHE STRING "Additional linker flags when linking unit tests.")
set(ROWPLUS_TEST_CUSTOM_CXX_FLAGS     "" CACHE STRING "Additional compiler flags when compiling unit tests.")

set(ROWPLUS_STANDARD_LIBRARIES_TO_LINK_TO "")

if(NOT STANDARD_MATH_LIBRARY_FOUND)

  message(FATAL_ERROR
      "Can't link to the standard math library. Please report to the ROWPlus developers, telling them about your platform.")

else()

  if(ROWPLUS_STANDARD_LIBRARIES_TO_LINK_TO)
    set(ROWPLUS_STANDARD_LIBRARIES_TO_LINK_TO "${ROWPLUS_STANDARD_LIBRARIES_TO_LINK_TO} ${STANDARD_MATH_LIBRARY}")
  else()
    set(ROWPLUS_STANDARD_LIBRARIES_TO_LINK_TO "${STANDARD_MATH_LIBRARY}")
  endif()

endif()

if(ROWPLUS_STANDARD_LIBRARIES_TO_LINK_TO)
  message(STATUS "Standard libraries to link to explicitly: ${ROWPLUS_STANDARD_LIBRARIES_TO_LINK_TO}")
else()
  message(STATUS "Standard libraries to link to explicitly: none")
endif()

#############################################################################
# find Eigen3
#############################################################################
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  message(FATAL_ERROR
      "Can't link to the Eigen3 library. Plesae specify ENV EIGEN3_ROOT.")
endif()

#############################################################################
# find Boost
#############################################################################
FIND_PACKAGE(Boost COMPONENTS timer REQUIRED)
IF (NOT Boost_FOUND)
  MESSAGE("Could not find the required C++ Boost libraries")
ENDIF (NOT Boost_FOUND)

#############################################################################
# compiler flags
#############################################################################
macro(rp_add_cxx_compiler_flag FLAG)
  string(REGEX REPLACE "-" "" SFLAG1 ${FLAG})
  string(REGEX REPLACE "\\+" "p" SFLAG ${SFLAG1})
  check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORT_${SFLAG})
  if(COMPILER_SUPPORT_${SFLAG})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endmacro(rp_add_cxx_compiler_flag)

rp_add_cxx_compiler_flag("-std=c++11")

#############################################################################
add_subdirectory(examples/Lorenz_96)
# Add chemistry examples
OPTION(${PROJECT_NAME}_ExpChemistry "Compile examples of chemical systems." ON)
IF(${PROJECT_NAME}_ExpChemistry)
  add_subdirectory(examples/ChemicalSystems)
ENDIF (${PROJECT_NAME}_ExpChemistry)