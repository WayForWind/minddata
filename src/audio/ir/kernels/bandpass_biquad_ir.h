

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_BANDPASS_BIQUAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_BANDPASS_BIQUAD_IR_H_

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
constexpr char kBandpassBiquadOperation[] = "BandpassBiquad";

class BandpassBiquadOperation : public TensorOperation {
 public:
  BandpassBiquadOperation(int32_t sample_rate, float central_freq, float Q, bool const_skirt_gain);

  ~BandpassBiquadOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kBandpassBiquadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
  bool const_skirt_gain_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_BANDPASS_BIQUAD_IR_H_
