# minddata

# Code Structure Overview

This repository demonstrates the core data processing pipeline and the implementation of various operators, divided into Python and C++ layers.

## Python Layer (`code/python/`)

- `/python/` defines the interfaces for invoking the pipeline and core operators.
- `/python/engine` contains the pipeline-related logic.
- `/python/audio`, `/python/text`, and `/python/vision` define the operator interfaces for different data modalities (audio, text, and vision respectively).
-  `/python/transforms` contains definitions of various operations designed to transform and process input data, including tensor type conversion and tensor padding.

## C++ Layer (`code/src/`)

- `/src/engine` contains the core logic of the data processing pipeline.
- `/src/api` includes interface definitions and encapsulation logic for data preprocessing in various modalities (audio, vision, and text).
- The actual operator implementations and intermediate layers are located in the corresponding directories under `/src/`. For example:`/src/audio/ir/kernels/` contains intermediate representations (IR) of audio-related operators.

## Python-C++ Binding and Runtime Utilities

- `code/src/api/python/pybind_conversion` provides a set of utilities for data conversion between Python and C++, such as converting a Python list of `DatasetNode` objects to a C++ `std::vector<std::shared_ptr<DatasetNode>>`.
- `code/src/api/python/python_mp` defines a multi-process runtime management framework that enables thread and worker process binding.

The complete code is at https://github.com/mindspore-ai/mindspore.
