
#include "OURSdata/dataset/audio/kernels/audio_utils.h"

#include "OURSdata/dataset/audio/kernels/pitch_shift_op.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// main function call for resample
Status PitchShiftOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return PitchShift(input, output, sample_rate_, n_steps_, bins_per_octave_, n_fft_, win_length_, hop_length_, window_);
}

Status PitchShiftOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("PitchShift", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
