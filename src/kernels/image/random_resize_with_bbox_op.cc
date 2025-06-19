

#include "OURSdata/dataset/kernels/image/random_resize_with_bbox_op.h"

#include "OURSdata/dataset/kernels/image/resize_with_bbox_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RandomResizeWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  // Randomly selects from the following four interpolation methods
  // 0-bilinear, 1-nearest_neighbor, 2-bicubic, 3-area
  IO_CHECK_VECTOR(input, output);
  auto interpolation = static_cast<InterpolationMode>(distribution_(random_generator_));
  std::shared_ptr<TensorOp> resize_with_bbox_op = std::make_shared<ResizeWithBBoxOp>(size1_, size2_, interpolation);
  RETURN_IF_NOT_OK(resize_with_bbox_op->Compute(input, output));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
