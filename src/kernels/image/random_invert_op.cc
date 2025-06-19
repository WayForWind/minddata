

#include "OURSdata/dataset/kernels/image/random_invert_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status RandomInvertOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input
  if (input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("RandomInvert: image shape is not <H,W,C>, got rank: " + std::to_string(input->Rank()));
  }
  if (input->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
    RETURN_STATUS_UNEXPECTED(
      "RandomInvert: image shape is incorrect, expected num of channels is 3, "
      "but got:" +
      std::to_string(input->shape()[kChannelIndexHWC]));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(input->type().AsCVType() != kCVInvalidType,
                               "RandomInvert: Cannot convert from OpenCV type, unknown CV type. Currently "
                               "supported data type: [int8, uint8, int16, uint16, int32, float16, float32, float64].");
  if (distribution_(random_generator_)) {
    return Invert(input, output);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
