

#include "OURSdata/dataset/kernels/image/adjust_gamma_op.h"

#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
constexpr float AdjustGammaOp::kGain = 1.0;

Status AdjustGammaOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  // typecast
  CHECK_FAIL_RETURN_UNEXPECTED(
    !input->type().IsString(),
    "AdjustGamma: input tensor type should be int, float or double, but got: " + input->type().ToString());

  if (input->type().IsFloat()) {
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return AdjustGamma(input_tensor, output, gamma_, gain_);
  } else {
    return AdjustGamma(input, output, gamma_, gain_);
  }
}
}  // namespace dataset
}  // namespace ours
