

#include "OURSdata/dataset/kernels/image/random_sharpness_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
/// constructor
RandomSharpnessOp::RandomSharpnessOp(float start_degree, float end_degree)
    : start_degree_(start_degree), end_degree_(end_degree) {}

/// main function call for random sharpness : Generate the random degrees
Status RandomSharpnessOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  float random_double = distribution_(random_generator_);
  /// get the degree sharpness range
  /// the way this op works (uniform distribution)
  /// assumption here is that mDegreesEnd > mDegreeStart so we always get positive number
  float degree_range = (end_degree_ - start_degree_) / 2;
  float mid = (end_degree_ + start_degree_) / 2;
  float alpha = mid + random_double * degree_range;
  return AdjustSharpness(input, output, alpha);
}
}  // namespace dataset
}  // namespace ours
