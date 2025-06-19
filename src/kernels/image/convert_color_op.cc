

#include "OURSdata/dataset/kernels/image/convert_color_op.h"

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
ConvertColorOp::ConvertColorOp(ConvertMode convert_mode) : convert_mode_(convert_mode) {}

Status ConvertColorOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return ConvertColor(input, output, convert_mode_);
}
}  // namespace dataset
}  // namespace ours
