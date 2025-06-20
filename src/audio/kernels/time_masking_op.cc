
#include "OURSdata/dataset/audio/kernels/time_masking_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// constructor
TimeMaskingOp::TimeMaskingOp(bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value)
    : iid_masks_(iid_masks), time_mask_param_(time_mask_param), mask_start_(mask_start), mask_value_(mask_value) {}

// main function
Status TimeMaskingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input <..., freq, time>
  RETURN_IF_NOT_OK(ValidateLowRank("TimeMasking", input, kDefaultAudioDim, "<..., freq, time>"));
  const int32_t kTimeIndex = -1;
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape()[kTimeIndex] >= time_mask_param_,
                               "TimeMasking: invalid parameter, 'time_mask_param' should be less than or equal to "
                               "the length of time dimension, but got: 'frequency_mask_param' " +
                                 std::to_string(time_mask_param_) + " and length " +
                                 std::to_string(input->shape()[kTimeIndex]));

  std::shared_ptr<Tensor> input_tensor;
  // typecast
  RETURN_IF_NOT_OK(ValidateTensorNumeric("TimeMasking", input));
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
  } else {
    input_tensor = input;
  }

  // iid_masks - whether to apply different masks to each example/channel.
  if (!iid_masks_) {
    return MaskAlongAxis(input_tensor, output, time_mask_param_, mask_start_, mask_value_, 2);
  } else {
    return RandomMaskAlongAxis(input_tensor, output, time_mask_param_, mask_value_, 2, &random_generator_);
  }
}

Status TimeMaskingOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("TimeMasking", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
