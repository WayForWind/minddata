
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_EMNIST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_EMNIST_OP_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/engine/datasetops/source/mnist_op.h"

namespace ours {
namespace dataset {
// Forward declares
template <typename T>
class Queue;

class EMnistOp : public MnistOp {
 public:
  // Constructor.
  // @param const std::string &name - Class of this dataset, can be
  //     "byclass","bymerge","balanced","letters","digits","mnist".
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test' or 'all'.
  // @param int32_t num_workers - Number of workers reading images in parallel.
  // @param const std::string &folder_path - Dir directory of emnist.
  // @param int32_t queue_size - Connector queue size.
  // @param std::unique_ptr<DataSchema> data_schema - The schema of the Emnist dataset.
  // @param std::shared_ptr<SamplerRT> sampler - Sampler tells EMnistOp what to read.
  EMnistOp(const std::string &name, const std::string &usage, int32_t num_workers, const std::string &folder_path,
           int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  // Destructor.
  ~EMnistOp() = default;

  // A print method typically used for debugging.
  // @param std::ostream &out - Out stream.
  // @param bool show_all - Whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the EMNIST dataset.
  // @param const std::string &dir - Path to the EMNIST directory.
  // @param const std::string &name - Class of this dataset, can be
  //     "byclass","bymerge","balanced","letters","digits","mnist".
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test' or 'all'.
  // @param int64_t *count - Output arg that will hold the minimum of the actual dataset size and numSamples.
  // @return Status The status code returned.
  static Status CountTotalRows(const std::string &dir, const std::string &name, const std::string &usage,
                               int64_t *count);

  // Op name getter.
  // @return Name of the current Op.
  std::string Name() const override { return "EMnistOp"; }

  // DatasetName name getter.
  // \return DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const override { return upper ? "EMnist" : "emnist"; }

 private:
  // Read all files in the directory.
  // @return Status The status code returned.
  Status WalkAllFiles() override;

  const std::string name_;  // can be "byclass", "bymerge", "balanced", "letters", "digits", "mnist".
};

}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_EMNIST_OP_H_
