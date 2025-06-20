file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=SubModuleId::SM_MD)
if(ENABLE_PYTHON)
    add_library(APItoPython OBJECT
            python/bindings/dataset/audio/bindings.cc
            python/bindings/dataset/audio/kernels/ir/bindings.cc
            python/bindings/dataset/callback/bindings.cc
            python/bindings/dataset/core/bindings.cc
            =
            python/bindings/dataset/engine/datasetops/bindings.cc
            python/bindings/dataset/engine/ir/consumer/bindings.cc
            python/bindings/dataset/engine/ir/datasetops/bindings.cc
            python/bindings/dataset/engine/ir/datasetops/source/bindings.cc
            python/bindings/dataset/engine/ir/datasetops/source/samplers/bindings.cc
            python/bindings/dataset/engine/ir/execute/bindings.cc
            python/bindings/dataset/engine/ir/schema/bindings.cc
            python/bindings/dataset/engine/perf/bindings.cc
            python/bindings/dataset/kernels/bindings.cc
            python/bindings/dataset/kernels/ir/bindings.cc
            python/bindings/dataset/kernels/ir/image/bindings.cc
            python/bindings/dataset/text/bindings.cc
            python/bindings/dataset/text/kernels/ir/bindings.cc
            python/bindings/dataset/util/bindings.cc

            python/pybind_conversion.cc
            python/pybind_register.cc
            )
    target_include_directories(APItoPython PRIVATE ${pybind11_INCLUDE_DIRS})
endif()

add_library(cpp-API OBJECT
        audio.cc
        config.cc
        data_helper.cc
        datasets.cc
        execute.cc
        iterator.cc
        transforms.cc
        samplers.cc
        text.cc
        vision.cc
        )
