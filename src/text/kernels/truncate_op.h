/
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TRUNCATE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TRUNCATE_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class TruncateOp : public TensorOp {
 public:
  explicit TruncateOp(int32_t max_seq_len) : max_seq_len_(max_seq_len) {}

  ~TruncateOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kTruncateOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 private:
  int32_t max_seq_len_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TRUNCATE_OP_H_
