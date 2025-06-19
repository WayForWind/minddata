
#include "OURSdata/dataset/audio/kernels/dc_shift_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"

namespace ours {
namespace dataset {
Status DCShiftOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input <..., time>.
  RETURN_IF_NOT_OK(ValidateLowRank("DCShift", input, kMinAudioDim, "<..., time>"));
  // If datatype is not a numeric type, then we cannot deal with the data.
  RETURN_IF_NOT_OK(ValidateTensorNumeric("DCShift", input));
  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return DCShift<double>(input, output, shift_, limiter_gain_);
  } else {
    std::shared_ptr<Tensor> tmp;
    RETURN_IF_NOT_OK(TypeCast(input, &tmp, DataType(DataType::DE_FLOAT32)));
    return DCShift<float>(tmp, output, shift_, limiter_gain_);
  }
}

Status DCShiftOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("DCShift", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
