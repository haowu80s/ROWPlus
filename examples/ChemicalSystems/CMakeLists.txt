INCLUDE(${CMAKE_MODULE_PATH}/ROWPlusCommonInclude.cmake)

set(SOURCE_FILES RxnFunctor.cpp FixedStepSize.cpp AdapStepSize.cpp)

add_library (RxnFunctor RxnFunctor.cpp)

add_executable(rxnFixedStepSize FixedStepSize.cpp)
TARGET_LINK_LIBRARIES(rxnFixedStepSize ${KIT_LIBS} RxnFunctor)

add_executable(rxnAdapStepSize AdapStepSize.cpp)
TARGET_LINK_LIBRARIES(rxnAdapStepSize ${KIT_LIBS} RxnFunctor)