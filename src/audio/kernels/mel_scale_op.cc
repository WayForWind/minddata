/
#include "OURSdata/dataset/audio/kernels/mel_scale_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status MelScaleOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // check and init
  IO_CHECK(input, output);

  // check input dimension, it should be greater than 0
  RETURN_IF_NOT_OK(ValidateLowRank("MelScale", input, kDefaultAudioDim, "<..., freq, time>"));

  // check input dimension, it should be equal with input n_stft
  TensorShape input_shape = input->shape();
  RETURN_IF_NOT_OK(ValidateEqual("MelScale", "n_stft", n_stft_, "freq", static_cast<int>(input_shape[-2])));

  // check input type, it should be [int, float, double]
  RETURN_IF_NOT_OK(ValidateTensorNumeric("MelScale", input));

  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return MelScale<float>(input_tensor, output, n_mels_, sample_rate_, f_min_, f_max_, n_stft_, norm_, mel_type_);
  } else {
    input_tensor = input;
    return MelScale<double>(input_tensor, output, n_mels_, sample_rate_, f_min_, f_max_, n_stft_, norm_, mel_type_);
  }
}

Status MelScaleOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  auto input_size = inputs[0].AsVector();
  input_size.pop_back();
  TensorShape out = TensorShape(input_size);
  outputs.emplace_back(out);
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "MelScale: invalid input shape.");
}

Status MelScaleOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("MelScale", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
