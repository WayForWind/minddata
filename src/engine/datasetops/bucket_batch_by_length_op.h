
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_BUCKET_BATCH_BY_LENGTH_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_BUCKET_BATCH_BY_LENGTH_OP_H_

#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/dataset_iterator.h"
#include "OURSdata/dataset/engine/datasetops/batch_op.h"
#include "OURSdata/dataset/engine/datasetops/pipeline_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class BucketBatchByLengthOp : public PipelineOp {
 public:
  BucketBatchByLengthOp(const std::vector<std::string> &length_dependent_columns,
                        const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
                        std::shared_ptr<TensorOp> element_length_function, const PadInfo &pad_info,
                        bool pad_to_bucket_boundary, bool drop_remainder, int32_t op_connector_size);

  // Destructor
  ~BucketBatchByLengthOp() = default;

  // Might need to batch remaining buckets after receiving eoe, so override this method.
  // @param int32_t workerId
  // @return Status The status code returned
  Status EoeReceived(int32_t) override;

  std::string Name() const override { return kBucketBatchByLengthOp; }

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param sO - reference to the BucketBatchByLengthOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const BucketBatchByLengthOp &bo) {
    bo.Print(out, false);
    return out;
  }

  // Main loop of batch
  // @return Status The status code returned
  Status operator()() override;

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  Status ObtainElementLength(int32_t *out_element_length, TensorRow element);

  Status PadAndBatchBucket(int32_t bucket_index, TensorRow *batched_bucket);

  Status ComputeColMap() override;

  std::vector<std::string> length_dependent_columns_;
  std::vector<int32_t> bucket_boundaries_;
  std::vector<int32_t> bucket_batch_sizes_;
  std::shared_ptr<TensorOp> element_length_function_;
  PadInfo pad_info_;
  bool pad_to_bucket_boundary_;
  bool drop_remainder_;
  bool eoe_received_ = false;

  int32_t batch_count_;
  std::unique_ptr<ChildIterator> child_iterator_;
  std::vector<std::unique_ptr<TensorQTable>> buckets_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_BUCKET_BATCH_BY_LENGTH_OP_H_
