

#include "OURSdata/dataset/kernels/image/random_lighting_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status RandomLightingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  // check input dimension, it should be greater than 2
  RETURN_IF_NOT_OK(ValidateLowRank("RandomLighting", input, kMinImageRank, "<height, width, ...>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("RandomLighting", input));

  float rnd_r = dist_(random_generator_);
  float rnd_g = dist_(random_generator_);
  float rnd_b = dist_(random_generator_);
  return RandomLighting(input, output, rnd_r, rnd_g, rnd_b);
}
}  // namespace dataset
}  // namespace ours
