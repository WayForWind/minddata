/

#include "OURSdata/dataset/audio/kernels/mel_spectrogram_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status MelSpectrogramOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return MelSpectrogram(input, output, sample_rate_, n_fft_, win_length_, hop_length_, f_min_, f_max_, pad_, n_mels_,
                        window_, power_, normalized_, center_, pad_mode_, onesided_, norm_, mel_scale_);
}

Status MelSpectrogramOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  auto output_shape_vector = inputs[0].AsVector();
  auto time = output_shape_vector[output_shape_vector.size()];
  output_shape_vector.pop_back();
  output_shape_vector.push_back(n_mels_);
  output_shape_vector.push_back(time);
  TensorShape out = TensorShape(output_shape_vector);
  (void)outputs.emplace_back(out);
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "MelSpectrogram: input tensor is not in shape of <..., time>.");
}

Status MelSpectrogramOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("MelSepctrogram", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
