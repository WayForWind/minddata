

#include "OURSdata/dataset/kernels/image/auto_contrast_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
const float AutoContrastOp::kCutOff = 0.0;
const std::vector<uint32_t> AutoContrastOp::kIgnore = {};

Status AutoContrastOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return AutoContrast(input, output, cutoff_, ignore_);
}
}  // namespace dataset
}  // namespace ours
