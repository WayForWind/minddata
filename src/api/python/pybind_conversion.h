

#ifndef OURS_CCSRC_OURSDATA_DATASET_API_PYTHON_PYBIND_CONVERSION_H_
#define OURS_CCSRC_OURSDATA_DATASET_API_PYTHON_PYBIND_CONVERSION_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "dataset/api/python/pybind_register.h"
#include "dataset/core/type_id.h"

#include "dataset/engine/ir/datasetops/source/csv_node.h"
#include "dataset/include/dataset/datasets.h"
#include "dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "dataset/engine/ir/datasetops/source/samplers/prebuilt_sampler_ir.h"
#include "dataset/kernels/ir/data/transforms_ir.h"
#include "dataset/kernels/py_func_op.h"
namespace py = pybind11;

namespace OURS {
namespace dataset {
float toFloat(const py::handle &handle);

int toInt(const py::handle &handle);

int64_t toInt64(const py::handle &handle);

bool toBool(const py::handle &handle);

std::string toString(const py::handle &handle);

std::set<std::string> toStringSet(const py::list &list);

std::map<std::string, int32_t> toStringMap(const py::dict &dict);

std::map<std::string, float> toStringFloatMap(const py::dict &dict);

std::vector<std::string> toStringVector(const py::list &list);

std::vector<pid_t> toIntVector(const py::list &input_list);

std::vector<int64_t> toInt64Vector(const py::list &input_list);

std::unordered_map<int32_t, std::vector<pid_t>> toIntMap(const py::dict &input_dict);

std::pair<int64_t, int64_t> toIntPair(const py::tuple &tuple);

std::vector<std::pair<int, int>> toPairVector(const py::list &list);

std::vector<std::shared_ptr<TensorOperation>> toTensorOperations(const py::list &operations);

std::shared_ptr<TensorOperation> toTensorOperation(const py::handle &operation);

std::vector<std::shared_ptr<DatasetNode>> toDatasetNode(const std::shared_ptr<DatasetNode> &self,
                                                        const py::list &datasets);

std::shared_ptr<SamplerObj> toSamplerObj(const py::handle &py_sampler, bool isMindDataset = false);



ShuffleMode toShuffleMode(int32_t shuffle);

std::vector<std::shared_ptr<CsvBase>> toCSVBase(const py::list &csv_bases);

std::shared_ptr<TensorOp> toPyFuncOp(const py::object &func, DataType::Type data_type);

Status ToJson(const py::handle &padded_sample, nlohmann::json *padded_sample_json,
              std::map<std::string, std::string> *sample_bytes);

Status toPadInfo(const py::dict &value,
                 std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> *pad_info);

py::list shapesToListOfShape(const std::vector<TensorShape> &shapes);

py::list typesToListOfType(const std::vector<DataType> &types);

Status toIntMapTensor(const py::dict &value, std::unordered_map<std::int16_t, std::shared_ptr<Tensor>> *feature);
}  // namespace dataset
}  // namespace OURS
#endif  // OURS_CCSRC_OURSDATA_DATASET_API_PYTHON_PYBIND_CONVERSION_H_
