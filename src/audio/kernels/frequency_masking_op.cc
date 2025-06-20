
#include "OURSdata/dataset/audio/kernels/frequency_masking_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// constructor
FrequencyMaskingOp::FrequencyMaskingOp(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start,
                                       float mask_value)
    : frequency_mask_param_(frequency_mask_param),
      mask_start_(mask_start),
      iid_masks_(iid_masks),
      mask_value_(mask_value) {}

// main function
Status FrequencyMaskingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input <..., freq, time>
  RETURN_IF_NOT_OK(ValidateLowRank("FrequencyMasking", input, kDefaultAudioDim, "<..., freq, time>"));
  const int32_t kFreqIndex = -2;
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->shape()[kFreqIndex] >= frequency_mask_param_,
    "FrequencyMasking: invalid parameter, 'frequency_mask_param' should be less than or equal to "
    "the length of frequency dimension, but got: 'frequency_mask_param' " +
      std::to_string(frequency_mask_param_) + " and length " + std::to_string(input->shape()[kFreqIndex]));

  std::shared_ptr<Tensor> input_tensor;
  // typecast
  RETURN_IF_NOT_OK(ValidateTensorNumeric("FrequencyMasking", input));
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
  } else {
    input_tensor = input;
  }
  // iid_masks - whether to apply different masks to each example/channel.
  if (!iid_masks_) {
    return MaskAlongAxis(input_tensor, output, frequency_mask_param_, mask_start_, mask_value_, 1);
  } else {
    return RandomMaskAlongAxis(input_tensor, output, frequency_mask_param_, mask_value_, 1, &random_generator_);
  }
}

Status FrequencyMaskingOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("FrequencyMasking", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
