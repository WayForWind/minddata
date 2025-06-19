
#include "OURSdata/dataset/audio/kernels/fade_op.h"

#include <cmath>

#include "OURSdata/dataset/audio/kernels/audio_utils.h"

namespace ours {
namespace dataset {
constexpr int32_t FadeOp::kFadeInLen = 0;
constexpr int32_t FadeOp::kFadeOutLen = 0;
constexpr FadeShape FadeOp::kFadeShape = FadeShape::kLinear;

Status FadeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("Fade", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Fade", input));
  if (fade_in_len_ == 0 && fade_out_len_ == 0) {
    *output = input;
  } else {
    RETURN_IF_NOT_OK(Fade(input, output, fade_in_len_, fade_out_len_, fade_shape_));
  }
  return Status::OK();
}

Status FadeOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Fade", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
