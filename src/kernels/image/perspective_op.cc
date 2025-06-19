/
#include "OURSdata/dataset/kernels/image/perspective_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
PerspectiveOp::PerspectiveOp(const std::vector<std::vector<int32_t>> &start_points,
                             const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation)
    : start_points_(start_points), end_points_(end_points), interpolation_(interpolation) {}

Status PerspectiveOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return Perspective(input, output, start_points_, end_points_, interpolation_);
}
}  // namespace dataset
}  // namespace ours
