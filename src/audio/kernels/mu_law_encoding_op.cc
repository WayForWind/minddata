
#include "OURSdata/dataset/audio/kernels/mu_law_encoding_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"

namespace ours {
namespace dataset {
// constructor
MuLawEncodingOp::MuLawEncodingOp(int32_t quantization_channels) : quantization_channels_(quantization_channels) {}

// main function
Status MuLawEncodingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("MuLawEncoding", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("MuLawEncoding", input));
  return MuLawEncoding(input, output, quantization_channels_);
}

Status MuLawEncodingOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("MuLawEncoding", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  outputs[0] = DataType(DataType::DE_INT32);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
