

#include "OURSdata/dataset/kernels/image/random_solarize_op.h"

#include <utility>

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RandomSolarizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  uint8_t threshold_min_ = threshold_[0], threshold_max_ = threshold_[1];

  CHECK_FAIL_RETURN_UNEXPECTED(threshold_min_ <= threshold_max_,
                               "RandomSolarize: min of threshold: " + std::to_string(threshold_min_) +
                                 " is greater than max of threshold: " + std::to_string(threshold_max_));

  float threshold_min = static_cast<float>(std::uniform_int_distribution(
    static_cast<uint32_t>(threshold_min_), static_cast<uint32_t>(threshold_max_))(random_generator_));
  float threshold_max = static_cast<float>(std::uniform_int_distribution(
    static_cast<uint32_t>(threshold_min_), static_cast<uint32_t>(threshold_max_))(random_generator_));

  if (threshold_max < threshold_min) {
    std::swap(threshold_min, threshold_max);
  }
  return Solarize(input, output, {threshold_min, threshold_max});
}
}  // namespace dataset
}  // namespace ours
