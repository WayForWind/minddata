file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=our::SubModuleId::SM_MD)

set(DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SRC_FILES
        distributed_sampler_ir.cc
        pk_sampler_ir.cc
        prebuilt_sampler_ir.cc
        random_sampler_ir.cc
        samplers_ir.cc
        sequential_sampler_ir.cc
        skip_first_epoch_sampler_ir.cc
        subset_random_sampler_ir.cc
        subset_sampler_ir.cc
        weighted_random_sampler_ir.cc

        )

add_library(engine-ir-datasetops-source-samplers OBJECT ${DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SRC_FILES})