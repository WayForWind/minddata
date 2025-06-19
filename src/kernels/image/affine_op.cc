
#include <random>
#include <vector>

#include "OURSdata/dataset/kernels/image/affine_op.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/image/math_utils.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
const InterpolationMode AffineOp::kDefInterpolation = InterpolationMode::kNearestNeighbour;
const float_t AffineOp::kDegrees = 0.0;
const std::vector<float_t> AffineOp::kTranslation = {0.0, 0.0};
const float_t AffineOp::kScale = 1.0;
const std::vector<float_t> AffineOp::kShear = {0.0, 0.0};
const std::vector<uint8_t> AffineOp::kFillValue = {0, 0, 0};

AffineOp::AffineOp(float_t degrees, const std::vector<float_t> &translation, float_t scale,
                   const std::vector<float_t> &shear, InterpolationMode interpolation,
                   const std::vector<uint8_t> &fill_value)
    : degrees_(degrees),
      translation_(translation),
      scale_(scale),
      shear_(shear),
      interpolation_(interpolation),
      fill_value_(fill_value) {}

Status AffineOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->shape().Size() == kMinImageRank || input->shape().Size() == kDefaultImageRank,
    "Affine: input tensor is not in shape of <H,W,C> or <H,W>, but got rank: " + std::to_string(input->shape().Size()));
  dsize_t height = input->shape()[0];
  dsize_t width = input->shape()[1];
  float_t translation_x = translation_[0] * static_cast<float>(width);
  float_t translation_y = translation_[1] * static_cast<float>(height);
  std::vector<float_t> new_translation{translation_x, translation_y};
  if (fill_value_.size() == 1) {
    fill_value_ = {fill_value_[0], fill_value_[0], fill_value_[0]};
  }
  return Affine(input, output, degrees_, new_translation, scale_, shear_, interpolation_, fill_value_);
}
}  // namespace dataset
}  // namespace ours
