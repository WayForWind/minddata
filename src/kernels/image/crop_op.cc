
#include "OURSdata/dataset/kernels/image/crop_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status CropOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImageRank("Crop", input->shape().Size()));
  auto input_h = static_cast<int>(input->shape()[0]);
  auto input_w = static_cast<int>(input->shape()[1]);
  CHECK_FAIL_RETURN_UNEXPECTED(y_ + height_ <= input_h, "Crop: Crop height dimension: " + std::to_string(y_ + height_) +
                                                          " exceeds image height: " + std::to_string(input_h));
  CHECK_FAIL_RETURN_UNEXPECTED(x_ + width_ <= input_w, "Crop: Crop width dimension: " + std::to_string(x_ + width_) +
                                                         " exceeds image width: " + std::to_string(input_w));
  return Crop(input, output, x_, y_, width_, height_);
}

Status CropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{height_, width_};
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "Crop: inputs cannot be empty.");
  if (inputs[0].Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][2]));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(),
                               "Crop: invalid input shape, expected 2D or 3D input, but got input dimension is:" +
                                 std::to_string(inputs[0].Rank()));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
