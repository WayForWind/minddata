
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FREQUENCY_MASKING_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FREQUENCY_MASKING_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kFrequencyMaskingOperation[] = "FrequencyMasking";

class FrequencyMaskingOperation : public TensorOperation {
 public:
  FrequencyMaskingOperation(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, float mask_value);

  ~FrequencyMaskingOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  bool iid_masks_;
  int32_t frequency_mask_param_;
  int32_t mask_start_;
  float mask_value_;
};  // class FrequencyMaskingOperation
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FREQUENCY_MASKING_IR_H_
