

#include "OURSdata/dataset/kernels/image/sharpness_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
const float SharpnessOp::kDefAlpha = 1.0;

Status SharpnessOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return AdjustSharpness(input, output, alpha_);
}
}  // namespace dataset
}  // namespace ours
