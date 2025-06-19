
#include "OURSdata/dataset/kernels/image/rgb_to_gray_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status RgbToGrayOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return RgbToGray(input, output);
}
}  // namespace dataset
}  // namespace ours
