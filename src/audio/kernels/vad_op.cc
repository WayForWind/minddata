
#include "OURSdata/dataset/audio/kernels/vad_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status VadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("Vad", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Vad", input));
  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return Vad<float>(input_tensor, output, sample_rate_, trigger_level_, trigger_time_, search_time_, allowed_gap_,
                      pre_trigger_time_, boot_time_, noise_up_time_, noise_down_time_, noise_reduction_amount_,
                      measure_freq_, measure_duration_, measure_smooth_time_, hp_filter_freq_, lp_filter_freq_,
                      hp_lifter_freq_, lp_lifter_freq_);
  } else {
    input_tensor = input;
    return Vad<double>(input_tensor, output, sample_rate_, trigger_level_, trigger_time_, search_time_, allowed_gap_,
                       pre_trigger_time_, boot_time_, noise_up_time_, noise_down_time_, noise_reduction_amount_,
                       measure_freq_, measure_duration_, measure_smooth_time_, hp_filter_freq_, lp_filter_freq_,
                       hp_lifter_freq_, lp_lifter_freq_);
  }

  return Status::OK();
}

Status VadOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Vad", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
