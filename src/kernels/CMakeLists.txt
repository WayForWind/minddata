add_subdirectory(image)
add_subdirectory(data)
add_subdirectory(ir)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=our::SubModuleId::SM_MD)


set(COMMON_TENSOR_OPS
        data/compose_op.cc
        data/random_apply_op.cc
        data/random_choice_op.cc
        tensor_op.cc
        plugin_op.cc
        )

if(ENABLE_PYTHON)
    add_library(kernels OBJECT
        ${COMMON_TENSOR_OPS}
        c_func_op.cc
        py_func_op.cc
        )
    target_include_directories(kernels PRIVATE ${pybind11_INCLUDE_DIRS})
else()
    add_library(kernels OBJECT
            ${COMMON_TENSOR_OPS})
endif()
