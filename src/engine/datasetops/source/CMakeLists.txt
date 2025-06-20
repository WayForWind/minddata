add_subdirectory(sampler)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=our::SubModuleId::SM_MD)

set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
    ag_news_op.cc
    album_op.cc
    amazon_review_op.cc
    caltech_op.cc
    celeba_op.cc
    cifar_op.cc
    cityscapes_op.cc
    clue_op.cc
    cmu_arctic_op.cc
    coco_op.cc
    conll2000_op.cc
    csv_op.cc
    dbpedia_op.cc
    div2k_op.cc
    emnist_op.cc
    en_wik9_op.cc
    fake_image_op.cc
    fashion_mnist_op.cc
    flickr_op.cc
    food101_op.cc
    gtzan_op.cc
    image_folder_op.cc
    imdb_op.cc
    iwslt_op.cc
    io_block.cc
    kitti_op.cc
    kmnist_op.cc
    lfw_op.cc
    libri_tts_op.cc
    lj_speech_op.cc
    lsun_op.cc
    mappable_leaf_op.cc
    mnist_op.cc
    multi30k_op.cc
    nonmappable_leaf_op.cc
    omniglot_op.cc
    penn_treebank_op.cc
    photo_tour_op.cc
    places365_op.cc
    qmnist_op.cc
    random_data_op.cc
    rendered_sst2_op.cc
    sbu_op.cc
    semeion_op.cc
    sogou_news_op.cc
    speech_commands_op.cc
    squad_op.cc
    sst2_op.cc
    stl10_op.cc
    sun397_op.cc
    tedlium_op.cc
    text_file_op.cc
    udpos_op.cc
    usps_op.cc
    wider_face_op.cc
    wiki_text_op.cc
    yahoo_answers_op.cc
    yelp_review_op.cc
    yes_no_op.cc
    )

set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
    ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES}

    tf_reader_op.cc
    )

if(ENABLE_PYTHON)
    set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
        ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
        generator_op.cc
        voc_op.cc
        manifest_op.cc
        )
endif()

add_library(engine-datasetops-source OBJECT ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES})
