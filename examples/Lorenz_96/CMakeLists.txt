INCLUDE(${CMAKE_MODULE_PATH}/ROWPlusCommonInclude.cmake)
set(SOURCE_FILES lorenz96.cpp main.cpp)
add_executable(lorenz96 ${SOURCE_FILES})