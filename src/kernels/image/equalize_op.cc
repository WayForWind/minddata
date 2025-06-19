
#include "OURSdata/dataset/kernels/image/equalize_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
// only supports RGB images
Status EqualizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return Equalize(input, output);
}
}  // namespace dataset
}  // namespace ours
