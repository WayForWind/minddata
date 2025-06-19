
#include "OURSdata/dataset/kernels/image/solarize_op.h"

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status SolarizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return Solarize(input, output, threshold_);
}
}  // namespace dataset
}  // namespace ours
