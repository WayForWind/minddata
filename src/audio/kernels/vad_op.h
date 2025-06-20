

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_VAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_VAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class VadOp : public TensorOp {
 public:
  VadOp(int32_t sample_rate, float trigger_level, float trigger_time, float search_time, float allowed_gap,
        float pre_trigger_time, float boot_time, float noise_up_time, float noise_down_time,
        float noise_reduction_amount, float measure_freq, float measure_duration, float measure_smooth_time,
        float hp_filter_freq, float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq)
      : sample_rate_(sample_rate),
        trigger_level_(trigger_level),
        trigger_time_(trigger_time),
        search_time_(search_time),
        allowed_gap_(allowed_gap),
        pre_trigger_time_(pre_trigger_time),
        boot_time_(boot_time),
        noise_up_time_(noise_up_time),
        noise_down_time_(noise_down_time),
        noise_reduction_amount_(noise_reduction_amount),
        measure_freq_(measure_freq),
        measure_duration_(measure_duration),
        measure_smooth_time_(measure_smooth_time),
        hp_filter_freq_(hp_filter_freq),
        lp_filter_freq_(lp_filter_freq),
        hp_lifter_freq_(hp_lifter_freq),
        lp_lifter_freq_(lp_lifter_freq) {}

  ~VadOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kVadOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t sample_rate_;
  float trigger_level_;
  float trigger_time_;
  float search_time_;
  float allowed_gap_;
  float pre_trigger_time_;
  float boot_time_;
  float noise_up_time_;
  float noise_down_time_;
  float noise_reduction_amount_;
  float measure_freq_;
  float measure_duration_;
  float measure_smooth_time_;
  float hp_filter_freq_;
  float lp_filter_freq_;
  float hp_lifter_freq_;
  float lp_lifter_freq_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_VAD_OP_H_
