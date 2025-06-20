/

#include "OURSdata/dataset/audio/kernels/mfcc_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status MFCCOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return MFCC(input, output, sample_rate_, n_mfcc_, dct_type_, log_mels_, n_fft_, win_length_, hop_length_, f_min_,
              f_max_, pad_, n_mels_, window_, power_, normalized_, center_, pad_mode_, onesided_, norm_, norm_M_,
              mel_scale_);
}

Status MFCCOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  auto output_shape_vector = inputs[0].AsVector();
  auto time = output_shape_vector[output_shape_vector.size()];
  output_shape_vector.pop_back();
  output_shape_vector.push_back(n_mfcc_);
  output_shape_vector.push_back(time);
  TensorShape out = TensorShape(output_shape_vector);
  (void)outputs.emplace_back(out);
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "MFCC: input tensor is not in shape of <..., time>.");
}

Status MFCCOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("MFCC", inputs[0].IsNumeric(), "[float]", inputs[0].ToString()));
  outputs[0] = DataType(DataType::DE_FLOAT32);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
