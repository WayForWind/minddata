add_subdirectory(data)
add_subdirectory(vision)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=our::SubModuleId::SM_MD)

set(DATASET_KERNELS_IR_SRC_FILES
        validators.cc
        )

add_library(kernels-ir OBJECT ${DATASET_KERNELS_IR_SRC_FILES})
