

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_LOWPASS_BIQUAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_LOWPASS_BIQUAD_IR_H_

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
// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kLowpassBiquadOperation[] = "LowpassBiquad";

class LowpassBiquadOperation : public TensorOperation {
 public:
  LowpassBiquadOperation(int32_t sample_rate, float cutoff_freq, float Q);

  ~LowpassBiquadOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kLowpassBiquadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  float cutoff_freq_;
  float Q_;
};  // class LowpassBiquad
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_LOWPASS_BIQUAD_IR_H_
