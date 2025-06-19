
#include "OURSdata/dataset/kernels/image/hwc_to_chw_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status HwcToChwOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return HwcToChw(input, output);
}

Status HwcToChwOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "HWC2CHW: inputs cannot be empty.");
  TensorShape image_shape = inputs[0];
  constexpr auto kDefaultImageRank = 3;
  if (image_shape.Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(TensorShape{image_shape[2], image_shape[0], image_shape[1]});
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    !outputs.empty(),
    "HWC2CHW: invalid input shape, expected 3D input, but got input dimension is:" + std::to_string(inputs[0].Rank()));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
