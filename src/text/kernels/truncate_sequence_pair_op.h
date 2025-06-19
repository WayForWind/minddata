
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TRUNCATE_SEQUENCE_PAIR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TRUNCATE_SEQUENCE_PAIR_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {

class TruncateSequencePairOp : public TensorOp {
 public:
  explicit TruncateSequencePairOp(dsize_t max_length) : max_length_(max_length) {}

  ~TruncateSequencePairOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kTruncateSequencePairOp; }

  // Unknown input size
  uint32_t NumInput() override { return 2; }

  // Unknown output size
  uint32_t NumOutput() override { return 2; }

 private:
  dsize_t max_length_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TRUNCATE_SEQUENCE_PAIR_OP_H_
