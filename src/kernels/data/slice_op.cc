

#include "OURSdata/dataset/kernels/data/slice_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
Status SliceOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return input->Slice(output, slice_options_);
}
}  // namespace dataset
}  // namespace ours
