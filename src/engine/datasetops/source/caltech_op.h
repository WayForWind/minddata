
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_CALTECH_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_CALTECH_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/data_schema.h"
#include "OURSdata/dataset/engine/datasetops/parallel_op.h"
#include "OURSdata/dataset/engine/datasetops/source/image_folder_op.h"
#include "OURSdata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace ours {
namespace dataset {
/// \brief Read Caltech256 Dataset.
class CaltechOp : public ImageFolderOp {
 public:
  /// \brief Constructor.
  /// \param[in] num_workers Num of workers reading images in parallel.
  /// \param[in] file_dir Directory of caltech dataset.
  /// \param[in] queue_size Connector queue size.
  /// \param[in] do_decode Whether to decode the raw data.
  /// \param[in] data_schema Data schema of caltech256 dataset.
  /// \param[in] sampler Sampler tells CaltechOp what to read.
  CaltechOp(int32_t num_workers, const std::string &file_dir, int32_t queue_size, bool do_decode,
            std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~CaltechOp() = default;

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "CaltechOp"; }

  /// \brief DatasetName name getter.
  /// \param[in] upper Whether the returned name begins with uppercase.
  /// \return DatasetName of the current Op.
  std::string DatasetName(bool upper = false) const { return upper ? "Caltech" : "caltech"; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_CALTECH_OP_H_
