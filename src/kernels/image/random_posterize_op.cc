

#include "OURSdata/dataset/kernels/image/random_posterize_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
RandomPosterizeOp::RandomPosterizeOp(const std::vector<uint8_t> &bit_range) : bit_range_(bit_range) {}

Status RandomPosterizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  uint8_t bits =
    (bit_range_[0] == bit_range_[1])
      ? bit_range_[0]
      : static_cast<uint8_t>(std::uniform_int_distribution<uint32_t>(bit_range_[0], bit_range_[1])(random_generator_));
  return Posterize(input, output, bits);
}
}  // namespace dataset
}  // namespace ours
