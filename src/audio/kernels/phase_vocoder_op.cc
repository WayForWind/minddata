/
#include "OURSdata/dataset/audio/kernels/phase_vocoder_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"

namespace ours {
namespace dataset {
Status PhaseVocoderOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return PhaseVocoder(input, output, rate_, phase_advance_);
}

Status PhaseVocoderOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  const int32_t kTimePos = -2;
  const int32_t kComplexDimSize = 2;
  for (auto s : inputs) {
    std::vector<dsize_t> s_vec = s.AsVector();
    s_vec.pop_back();
    s_vec.pop_back();
    s_vec.push_back(std::ceil(s[kTimePos] / rate_));
    s_vec.push_back(kComplexDimSize);
    outputs.emplace_back(TensorShape(s_vec));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(), "PhaseVocoder: invalid shape of input tensor.");
  return Status::OK();
}

Status PhaseVocoderOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("PhaseVocoder", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
