/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PHASE_VOCODER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PHASE_VOCODER_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kPhaseVocoderOperation[] = "PhaseVocoder";

class PhaseVocoderOperation : public TensorOperation {
 public:
  PhaseVocoderOperation(float rate, const std::shared_ptr<Tensor> &phase_advance);

  ~PhaseVocoderOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  float rate_;
  std::shared_ptr<Tensor> phase_advance_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PHASE_VOCODER_IR_H_
