

#include "OURSdata/dataset/audio/kernels/db_to_amplitude_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status DBToAmplitudeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("DBToAmplitude", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("DBToAmplitude", input));

  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType(DataType::DE_FLOAT64)) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return DBToAmplitude<float>(input_tensor, output, ref_, power_);
  } else {
    input_tensor = input;
    return DBToAmplitude<double>(input_tensor, output, ref_, power_);
  }
}
}  // namespace dataset
}  // namespace ours
