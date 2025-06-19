

#include "OURSdata/dataset/kernels/image/random_auto_contrast_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status RandomAutoContrastOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Check input
  if (input->Rank() != kMinImageRank && input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("RandomAutoContrast: image shape is not <H,W,C> or <H,W>, got rank: " +
                             std::to_string(input->Rank()));
  }
  if (input->Rank() == kDefaultImageRank) {
    if (input->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED(
        "RandomAutoContrast: image shape is incorrect, expected num of channels is 3, "
        "but got: " +
        std::to_string(input->shape()[kChannelIndexHWC]));
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(input->type().AsCVType() != kCVInvalidType,
                               "RandomAutoContrast: Cannot convert from OpenCV type, unknown CV type. Currently "
                               "supported data type: [int8, uint8, int16, uint16, int32, float16, float32, float64].");
  if (distribution_(random_generator_)) {
    return AutoContrast(input, output, cutoff_, ignore_);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
