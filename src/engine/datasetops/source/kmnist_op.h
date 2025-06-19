
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_KMNIST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_KMNIST_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "OURSdata/dataset/engine/datasetops/source/mnist_op.h"

namespace ours {
namespace dataset {
/// \brief Forward declares.
template <typename T>
class Queue;

class KMnistOp : public MnistOp {
 public:
  /// \brief Constructor.
  /// \param[in] usage Usage of this dataset, can be 'train', 'test' or 'all'.
  /// \param[in] num_workers Number of workers reading images in parallel.
  /// \param[in] folder_path Dir directory of kmnist.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] data_schema The schema of the kmnist dataset.
  /// \param[in] Sampler Tells KMnistOp what to read.
  KMnistOp(const std::string &usage, int32_t num_workers, const std::string &folder_path, int32_t queue_size,
           std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~KMnistOp() = default;

  /// \brief Function to count the number of samples in the KMNIST dataset.
  /// \param[in] dir Path to the KMNIST directory.
  /// \param[in] usage Usage of this dataset, can be 'train', 'test' or 'all'.
  /// \param[in] count Output arg that will hold the minimum of the actual dataset size and numSamples.
  /// \return Status The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "KMnistOp"; }

  /// \brief Dataset name getter.
  /// \param[in] upper Whether to get upper name.
  /// \return Dataset name of the current Op.
  std::string DatasetName(bool upper = false) const override { return upper ? "KMnist" : "kmnist"; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_KMNIST_OP_H_
