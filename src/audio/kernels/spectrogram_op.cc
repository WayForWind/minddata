
#include "OURSdata/dataset/audio/kernels/spectrogram_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status SpectrogramOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return Spectrogram(input, output, pad_, window_, n_fft_, hop_length_, win_length_, power_, normalized_, center_,
                     pad_mode_, onesided_);
}

Status SpectrogramOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  constexpr int two = 2;
  int length_ = inputs[0][-1] + pad_ * 2 + n_fft_;
  int n_columns = 0;
  CHECK_FAIL_RETURN_UNEXPECTED(hop_length_ != 0, "Spectrogram: hop_length can not be zero.");
  while ((1 + n_columns++) * hop_length_ + n_fft_ <= length_) {
  }
  auto vec = inputs[0].AsVector();
  vec.pop_back();
  if (onesided_) {
    vec.push_back(n_fft_ / two + 1);
  } else {
    vec.push_back(n_fft_);
  }
  vec.push_back(n_columns);
  if (std::fabs(power_) <= std::numeric_limits<float>::epsilon()) {
    vec.push_back(two);
  }
  (void)outputs.emplace_back(TensorShape(vec));

  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "Spectrogram: input tensor is not in shape of <..., time>.");
}
}  // namespace dataset
}  // namespace ours
