

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_TREBLE_BIQUAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_TREBLE_BIQUAD_IR_H_

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

constexpr char kTrebleBiquadOperation[] = "TrebleBiquad";

class TrebleBiquadOperation : public TensorOperation {
 public:
  TrebleBiquadOperation(int32_t sample_rate, float gain, float central_freq, float Q);

  ~TrebleBiquadOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kTrebleBiquadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  float gain_;
  float central_freq_;
  float Q_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_TREBLE_BIQUAD_IR_H_
