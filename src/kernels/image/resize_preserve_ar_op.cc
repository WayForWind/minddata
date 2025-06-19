
#include "OURSdata/dataset/kernels/image/resize_preserve_ar_op.h"

#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
const int32_t ResizePreserveAROp::kDefImgOrientation = 0;

ResizePreserveAROp::ResizePreserveAROp(int32_t height, int32_t width, int32_t img_orientation)
    : height_(height), width_(width), img_orientation_(img_orientation) {}

Status ResizePreserveAROp::Compute(const TensorRow &inputs, TensorRow *outputs) {
  IO_CHECK_VECTOR(inputs, outputs);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
