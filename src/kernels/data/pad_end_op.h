
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_PAD_END_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_PAD_END_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class PadEndOp : public TensorOp {
 public:
  PadEndOp(const TensorShape &pad_shape, std::shared_ptr<Tensor> pad_value)
      : output_shape_(pad_shape), pad_val_(std::move(pad_value)) {}

  ~PadEndOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kPadEndOp; }

 private:
  TensorShape output_shape_;
  std::shared_ptr<Tensor> pad_val_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_PAD_END_OP_H_
