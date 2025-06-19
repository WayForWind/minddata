
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_CONSUMERS_PYTHON_TREE_CONSUMER_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_CONSUMERS_PYTHON_TREE_CONSUMER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "pybind11/pybind11.h"

#include "OURSdata/dataset/engine/consumers/pull_based_tree_consumer.h"
#include "OURSdata/dataset/engine/consumers/tree_consumer.h"

namespace ours::dataset {

/// Consumer that iterates over the dataset and returns the rows one by one as a python list or a dict

class PythonIteratorConsumer : public IteratorConsumer {
 public:
  /// Constructor which will call the base class default constructor.
  /// \param num_epochs number of epochs. Default to -1 (infinite epochs).
  explicit PythonIteratorConsumer(int32_t num_epochs = -1) : IteratorConsumer(num_epochs) {}

  ~PythonIteratorConsumer() = default;

  Status Init(const std::shared_ptr<DatasetNode> &root, int64_t global_step = 0, int64_t dataset_size = -1) override;

  /// Returns the next row in a vector format
  /// \param[out] out std::vector of Tensors
  /// \return Status error code
  Status GetNextAsList(py::list *out);

  /// Returns the next row in as a map
  /// \param[out] out std::map of string to Tensor
  /// \return Status error code
  Status GetNextAsDict(const py::dict *out);
};

class PythonPullBasedIteratorConsumer : public PullBasedIteratorConsumer {
 public:
  /// Constructor which will call the base class default constructor.
  /// \param num_epochs number of epochs. Default to -1 (infinite epochs).
  explicit PythonPullBasedIteratorConsumer(int32_t num_epochs = -1) : PullBasedIteratorConsumer(num_epochs) {}

  ~PythonPullBasedIteratorConsumer() = default;

  /// Returns the next row in a vector format
  /// \param[out] out std::vector of Tensors
  /// \return Status error code
  Status GetNextAsList(py::list *out);

  /// Returns the next row in as a map
  /// \param[out] out std::map of string to Tensor
  /// \return Status error code
  Status GetNextAsDict(const py::dict *out);
};

class PythonBuildVocabConsumer : public BuildVocabConsumer {
 public:
  Status Start() override;
};

class PythonSaveToDisk : public SaveToDisk {
 public:
  PythonSaveToDisk(const std::string &datasetPath, int32_t numFiles, const std::string &datasetType);

  ~PythonSaveToDisk() = default;

  Status Save() override;
};

class PythonTreeGetters : public TreeGetters {
 public:
  Status GetRow(TensorRow *const r) override;

  ~PythonTreeGetters() = default;
};

class PythonDatasetSizeGetter : public DatasetSizeGetter {
 public:
  Status GetRow(const std::shared_ptr<TreeAdapter> &tree_adapter, TensorRow *r) override;

  ~PythonDatasetSizeGetter() = default;
};
}  // namespace ours::dataset
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_CONSUMERS_PYTHON_TREE_CONSUMER_H_
