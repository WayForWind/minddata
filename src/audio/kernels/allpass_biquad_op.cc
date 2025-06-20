
#include "OURSdata/dataset/audio/kernels/allpass_biquad_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status AllpassBiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("AllpassBiquad", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorFloat("AllpassBiquad", input));
  double w0 = 2 * PI * central_freq_ / sample_rate_;
  double alpha = sin(w0) / 2 / Q_;
  double b0 = 1 - alpha;
  double b1 = -2 * cos(w0);
  double b2 = 1 + alpha;
  double a0 = b2;
  double a1 = -2 * cos(w0);
  double a2 = 1 - alpha;
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
