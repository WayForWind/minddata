

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DETECT_PITCH_FREQUENCY_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DETECT_PITCH_FREQUENCY_IR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kDetectPitchFrequencyOperation[] = "DetectPitchFrequency";

class DetectPitchFrequencyOperation : public TensorOperation {
 public:
  DetectPitchFrequencyOperation(int32_t sample_rate, float frame_time, int32_t win_length, int32_t freq_low,
                                int32_t freq_high);

  ~DetectPitchFrequencyOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDetectPitchFrequencyOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  float frame_time_;
  int32_t win_length_;
  int32_t freq_low_;
  int32_t freq_high_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DETECT_PITCH_FREQUENCY_IR_H_
