/

#include "OURSdata/dataset/kernels/image/adjust_saturation_op.h"

#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status AdjustSaturationOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  return AdjustSaturation(input, output, saturation_factor_);
}
}  // namespace dataset
}  // namespace ours
