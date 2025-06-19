
#include "OURSdata/dataset/kernels/image/gaussian_blur_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status GaussianBlurOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImageRank("GaussianBlur", input->Rank()));
  return GaussianBlur(input, output, kernel_x_, kernel_y_, sigma_x_, sigma_y_);
}
}  // namespace dataset
}  // namespace ours
