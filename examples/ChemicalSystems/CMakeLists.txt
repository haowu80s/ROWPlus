INCLUDE(${CMAKE_MODULE_PATH}/ROWPlusCommonInclude.cmake)

set(SOURCE_FILES RxnFunctor.cpp main.cpp)

add_executable(reaction ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(reaction ${KIT_LIBS})