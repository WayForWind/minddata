

#include "OURSdata/dataset/audio/ir/kernels/frequency_masking_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/frequency_masking_op.h"

namespace ours {
namespace dataset {
namespace audio {
FrequencyMaskingOperation::FrequencyMaskingOperation(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start,
                                                     float mask_value)
    : iid_masks_(iid_masks),
      frequency_mask_param_(frequency_mask_param),
      mask_start_(mask_start),
      mask_value_(mask_value) {}

FrequencyMaskingOperation::~FrequencyMaskingOperation() = default;

Status FrequencyMaskingOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("FrequencyMasking", "frequency_mask_param", frequency_mask_param_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("FrequencyMasking", "mask_start", mask_start_));

  return Status::OK();
}

std::shared_ptr<TensorOp> FrequencyMaskingOperation::Build() {
  std::shared_ptr<FrequencyMaskingOp> tensor_op =
    std::make_shared<FrequencyMaskingOp>(iid_masks_, frequency_mask_param_, mask_start_, mask_value_);
  return tensor_op;
}

std::string FrequencyMaskingOperation::Name() const { return kFrequencyMaskingOperation; }

Status FrequencyMaskingOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["frequency_mask_param"] = frequency_mask_param_;
  args["mask_start"] = mask_start_;
  args["iid_masks"] = iid_masks_;
  args["mask_value"] = mask_value_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
