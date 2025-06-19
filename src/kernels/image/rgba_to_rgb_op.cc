
#include "OURSdata/dataset/kernels/image/rgba_to_rgb_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RgbaToRgbOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return RgbaToRgb(input, output);
}
}  // namespace dataset
}  // namespace ours
