

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_VAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_VAD_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kVadOperation[] = "Vad";

class VadOperation : public TensorOperation {
 public:
  VadOperation(int32_t sample_rate, float trigger_level, float trigger_time, float search_time, float allowed_gap,
               float pre_trigger_time, float boot_time, float noise_up_time, float noise_down_time,
               float noise_reduction_amount, float measure_freq, float measure_duration, float measure_smooth_time,
               float hp_filter_freq, float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq);

  ~VadOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

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
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_VAD_IR_H_
