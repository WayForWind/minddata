add_subdirectory(ir)
add_subdirectory(kernels)

file(GLOB _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=our::SubModuleId::SM_MD)
add_library(text OBJECT
        char_n_gram.cc
        fast_text.cc
        glove.cc
        sentence_piece_vocab.cc
        vectors.cc
        vocab.cc
        )

add_dependencies(text text-kernels)
