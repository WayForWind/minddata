
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_YES_NO_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_YES_NO_OP_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/data_schema.h"
#include "OURSdata/dataset/engine/datasetops/parallel_op.h"
#include "OURSdata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "OURSdata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "OURSdata/dataset/util/path.h"
#include "OURSdata/dataset/util/queue.h"
#include "OURSdata/dataset/util/services.h"
#include "OURSdata/dataset/util/status.h"
#include "OURSdata/dataset/util/wait_post.h"

namespace ours {
namespace dataset {
class YesNoOp : public MappableLeafOp {
 public:
  /// Constructor.
  /// @param std::string file_dir - dir directory of YesNo.
  /// @param int32_t num_workers - number of workers reading images in parallel.
  /// @param int32_t queue_size - connector queue size.
  /// @param std::unique_ptr<DataSchema> data_schema - the schema of the YesNo dataset.
  /// @param std::shared_ptr<Sampler> sampler - sampler tells YesNoOp what to read.
  YesNoOp(const std::string &file_dir, int32_t num_workers, int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
          std::shared_ptr<SamplerRT> sampler);

  /// Destructor.
  ~YesNoOp() = default;

  /// A print method typically used for debugging.
  /// @param std::ostream &out - out stream.
  /// @param bool show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  /// Op name getter.
  /// @return Name of the current Op.
  std::string Name() const override { return "YesNoOp"; }

  /// @param int64_t *count - output rows number of YesNoDataset.
  /// @return Status - The status code returned.
  Status CountTotalRows(int64_t *count);

 private:
  /// Load a tensor row according to wave id.
  /// @param row_id_type row_id - id for this tensor row.
  /// @param TensorRow trow - wave & target read into this tensor row.
  /// @return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *trow) override;

  /// Get file infos by file name.
  /// @param string line - file name.
  /// @param vector split_num - vector of annotation.
  /// @return Status - The status code returned.
  Status Split(const std::string &line, std::vector<int32_t> *split_num);

  /// Initialize YesNoDataset related var, calls the function to walk all files.
  /// @return Status - The status code returned.
  Status PrepareData();

  /// Private function for computing the assignment of the column name map.
  /// @return Status - The status code returned.
  Status ComputeColMap() override;

  std::vector<std::string> all_wave_files_;
  std::string dataset_dir_;
  std::unique_ptr<DataSchema> data_schema_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_YES_NO_OP_H
