
#include "OURSdata/dataset/kernels/image/erase_op.h"

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// constructor
EraseOp::EraseOp(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<float> &value,
                 bool inplace)
    : top_(top), left_(left), height_(height), width_(width), value_(value), inplace_(inplace) {}

Status EraseOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImageDtype("Erase", input->type()));
  if (input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("Erase: input tensor is not in shape of <H,W,C>, but got rank: " +
                             std::to_string(input->Rank()));
  }
  int num_channels = input->shape()[2];
  if (num_channels != kDefaultImageChannel) {
    RETURN_STATUS_UNEXPECTED("Erase: channel of input image should be 3, but got: " + std::to_string(num_channels));
  }
  return Erase(input, output, top_, left_, height_, width_, value_, inplace_);
}
}  // namespace dataset
}  // namespace ours
