

#include "OURSdata/dataset/kernels/image/posterize_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
PosterizeOp::PosterizeOp(uint8_t bit) : bit_(bit) {}

Status PosterizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return Posterize(input, output, bit_);
}
}  // namespace dataset
}  // namespace ours
