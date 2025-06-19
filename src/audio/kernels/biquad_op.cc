
#include "OURSdata/dataset/audio/kernels/biquad_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status BiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("Biquad", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT32 or DE_FLOAT16 or DE_FLOAT64
  RETURN_IF_NOT_OK(ValidateTensorFloat("Biquad", input));
  if (input->type() == DataType(DataType::DE_FLOAT32)) {
    return Biquad(input, output, static_cast<float>(b0_), static_cast<float>(b1_), static_cast<float>(b2_),
                  static_cast<float>(a0_), static_cast<float>(a1_), static_cast<float>(a2_));
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Biquad(input, output, static_cast<double>(b0_), static_cast<double>(b1_), static_cast<double>(b2_),
                  static_cast<double>(a0_), static_cast<double>(a1_), static_cast<double>(a2_));
  } else {
    return Biquad(input, output, static_cast<float16>(b0_), static_cast<float16>(b1_), static_cast<float16>(b2_),
                  static_cast<float16>(a0_), static_cast<float16>(a1_), static_cast<float16>(a2_));
  }
}
}  // namespace dataset
}  // namespace ours
