

#include "OURSdata/dataset/kernels/image/random_adjust_sharpness_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status RandomAdjustSharpnessOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  // Check input
  if (input->Rank() != kMinImageRank && input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("RandomAdjustSharpness: image shape is not <H,W,C> or <H,W>, got rank: " +
                             std::to_string(input->Rank()));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(input->type().AsCVType() != kCVInvalidType,
                               "RandomAdjustSharpness: Cannot convert from OpenCV type, unknown CV type. Currently "
                               "supported data type: [int8, uint8, int16, uint16, int32, float16, float32, float64].");

  if (distribution_(random_generator_)) {
    return AdjustSharpness(input, output, degree_);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
