

#include "OURSdata/dataset/audio/kernels/gain_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status GainOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("Gain", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Gain", input));
  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Gain<double>(input, output, gain_db_);
  } else {
    std::shared_ptr<Tensor> float_input;
    RETURN_IF_NOT_OK(TypeCast(input, &float_input, DataType(DataType::DE_FLOAT32)));
    return Gain<float>(float_input, output, gain_db_);
  }
}
}  // namespace dataset
}  // namespace ours
