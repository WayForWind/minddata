

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_EQUALIZER_BIQUAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_EQUALIZER_BIQUAD_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {

constexpr char kEqualizerBiquadOperation[] = "EqualizerBiquad";

class EqualizerBiquadOperation : public TensorOperation {
 public:
  EqualizerBiquadOperation(int32_t sample_rate, float center_freq, float gain, float Q);

  ~EqualizerBiquadOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kEqualizerBiquadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  float center_freq_;
  float gain_;
  float Q_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_EQUALIZER_BIQUAD_IR_H_
