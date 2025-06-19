
#include "OURSdata/dataset/audio/kernels/detect_pitch_frequency_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status DetectPitchFrequencyOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("DetectPitchFrequency", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT16, DE_FLOAT32 or DE_FLOAT64
  RETURN_IF_NOT_OK(ValidateTensorFloat("DetectPitchFrequency", input));
  return DetectPitchFrequency(input, output, sample_rate_, frame_time_, win_length_, freq_low_, freq_high_);
}
}  // namespace dataset
}  // namespace ours
