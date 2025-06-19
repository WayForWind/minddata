/

#include "OURSdata/dataset/kernels/image/adjust_hue_op.h"

#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status AdjustHueOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  return AdjustHue(input, output, hue_factor_);
}
}  // namespace dataset
}  // namespace ours
