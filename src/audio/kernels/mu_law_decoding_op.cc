
#include "OURSdata/dataset/audio/kernels/mu_law_decoding_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"

namespace ours {
namespace dataset {
// constructor
MuLawDecodingOp::MuLawDecodingOp(int32_t quantization_channels) : quantization_channels_(quantization_channels) {}

// main function
Status MuLawDecodingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("MuLawDecoding", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("MuLawDecoding", input));
  return MuLawDecoding(input, output, quantization_channels_);
}

Status MuLawDecodingOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("MuLawDecoding", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
