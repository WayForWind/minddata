
#include "OURSdata/dataset/audio/kernels/bandpass_biquad_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status BandpassBiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("BandpassBiquad", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT32 or DE_FLOAT16 or DE_FLOAT64
  RETURN_IF_NOT_OK(ValidateTensorFloat("BandpassBiquad", input));
  float w0 = 2 * PI * central_freq_ / sample_rate_;
  float alpha = sin(w0) / 2 / Q_;
  float temp;
  if (const_skirt_gain_) {
    temp = sin(w0) / 2;
  } else {
    temp = alpha;
  }

  float b0 = temp;
  float b1 = 0.0;
  float b2 = -temp;
  float a0 = 1 + alpha;
  float a1 = (-2) * cos(w0);
  float a2 = 1 - alpha;

  if (input->type() == DataType(DataType::DE_FLOAT32)) {
    return Biquad(input, output, static_cast<float>(b0), static_cast<float>(b1), static_cast<float>(b2),
                  static_cast<float>(a0), static_cast<float>(a1), static_cast<float>(a2));
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Biquad(input, output, static_cast<double>(b0), static_cast<double>(b1), static_cast<double>(b2),
                  static_cast<double>(a0), static_cast<double>(a1), static_cast<double>(a2));
  } else {
    return Biquad(input, output, static_cast<float16>(b0), static_cast<float16>(b1), static_cast<float16>(b2),
                  static_cast<float16>(a0), static_cast<float16>(a1), static_cast<float16>(a2));
  }
}
}  // namespace dataset
}  // namespace ours
