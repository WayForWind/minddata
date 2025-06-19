
#include "OURSdata/dataset/kernels/data/pad_end_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
Status PadEndOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return PadEnd(input, output, output_shape_.AsVector(), pad_val_);
}

Status PadEndOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  for (auto i = 0; i < inputs.size(); ++i) {
    (void)outputs.emplace_back(TensorShape(output_shape_.AsVector()));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(), "PadEnd: invalid input shape.");
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
