/

#include "OURSdata/dataset/audio/kernels/inverse_spectrogram_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status InverseSpectrogramOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return InverseSpectrogram(input, output, length_, n_fft_, win_length_, hop_length_, pad_, window_, normalized_,
                            center_, pad_mode_, onesided_);
}

Status InverseSpectrogramOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  auto output_shape_vector = inputs[0].AsVector();
  auto time = output_shape_vector[output_shape_vector.size()];
  output_shape_vector.pop_back();
  output_shape_vector.pop_back();
  output_shape_vector.pop_back();
  output_shape_vector.push_back(time);
  TensorShape out = TensorShape(output_shape_vector);
  (void)outputs.emplace_back(out);
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError,
                "InverseSpectrogram: input tensor is not in shape of <..., freq, time, complex=2>.");
}

Status InverseSpectrogramOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("InverseSpectrogram", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
