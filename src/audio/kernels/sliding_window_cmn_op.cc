
#include "OURSdata/dataset/audio/kernels/sliding_window_cmn_op.h"

namespace ours {
namespace dataset {
Status SlidingWindowCmnOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(SlidingWindowCmn(input, output, cmn_window_, min_cmn_window_, center_, norm_vars_));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
