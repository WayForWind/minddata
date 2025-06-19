/

#include "OURSdata/dataset/audio/kernels/mask_along_axis_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"

namespace ours {
namespace dataset {
Status MaskAlongAxisOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input <..., freq, time>
  RETURN_IF_NOT_OK(ValidateLowRank("MaskAlongAxis", input, kDefaultAudioDim, "<..., freq, time>"));
  RETURN_IF_NOT_OK(
    ValidateTensorType("MaskAlongAxis", input->type().IsNumeric(), "[int, float, double]", input->type().ToString()));
  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
  } else {
    input_tensor = input;
  }
  return MaskAlongAxis(input_tensor, output, mask_width_, mask_start_, mask_value_, axis_);
}

Status MaskAlongAxisOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("MaskAlongAxis", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
