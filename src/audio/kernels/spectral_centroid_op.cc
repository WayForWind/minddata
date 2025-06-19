
#include "OURSdata/dataset/audio/kernels/spectral_centroid_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status SpectralCentroidOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return SpectralCentroid(input, output, sample_rate_, n_fft_, win_length_, hop_length_, pad_, window_);
}

Status SpectralCentroidOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("SpectralCentroid", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
