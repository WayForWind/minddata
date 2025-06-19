
#include "OURSdata/dataset/kernels/image/rgba_to_bgr_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RgbaToBgrOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return RgbaToBgr(input, output);
}
}  // namespace dataset
}  // namespace ours
