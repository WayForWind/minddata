

#include "OURSdata/dataset/audio/kernels/dither_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status DitherOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input dimension, it should be greater than 0
  RETURN_IF_NOT_OK(ValidateLowRank("Dither", input, kMinAudioDim, "<..., time>"));

  // check input type, it should be [int, float, double]
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Dither", input));

  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Dither<double>(input, output, density_function_, noise_shaping_, &random_generator_);
  } else {
    std::shared_ptr<Tensor> float_input;
    RETURN_IF_NOT_OK(TypeCast(input, &float_input, DataType(DataType::DE_FLOAT32)));
    return Dither<float>(float_input, output, density_function_, noise_shaping_, &random_generator_);
  }
}

Status DitherOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Dither", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
