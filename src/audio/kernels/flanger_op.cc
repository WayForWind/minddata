
#include "OURSdata/dataset/audio/kernels/flanger_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status FlangerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input dimensions, it should be 2 dimensions or more
  RETURN_IF_NOT_OK(ValidateLowRank("Flanger", input, kDefaultAudioDim, "<..., channel, time>"));

  // check input channel, it should be less than or equal to 4
  const int32_t kChannelIndex = -2;
  const int32_t kChannelLimit = 4;
  CHECK_FAIL_RETURN_SYNTAX_ERROR(input->shape()[kChannelIndex] <= kChannelLimit,
                                 "Flanger: the channel of input tensor dose not match the requirement of operator. "
                                 "Expecting tensor with channel less than or equal to 4. But got channel: " +
                                   std::to_string(input->shape()[kChannelIndex]));

  // check input type, it should be [int, float, double]
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Flanger", input));

  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Flanger<double>(input, output, sample_rate_, delay_, depth_, regen_, width_, speed_, phase_, Modulation_,
                           Interpolation_);
  } else {
    return Flanger<float>(input, output, sample_rate_, delay_, depth_, regen_, width_, speed_, phase_, Modulation_,
                          Interpolation_);
  }
}

Status FlangerOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Flanger", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  outputs[0] = inputs[0];
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
