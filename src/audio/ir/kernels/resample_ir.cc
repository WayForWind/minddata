

#include "OURSdata/dataset/audio/ir/kernels/resample_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/resample_op.h"

namespace ours {
namespace dataset {
namespace audio {
ResampleOperation::ResampleOperation(float orig_freq, float new_freq, ResampleMethod resample_method,
                                     int32_t lowpass_filter_width, float rolloff, float beta)
    : orig_freq_(orig_freq),
      new_freq_(new_freq),
      resample_method_(resample_method),
      lowpass_filter_width_(lowpass_filter_width),
      rolloff_(rolloff),
      beta_(beta) {}

std::string ResampleOperation::Name() const { return kResampleOperation; }

Status ResampleOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Resample", "orig_freq", orig_freq_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Resample", "new_freq", new_freq_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("Resample", "lowpass_filter_width", lowpass_filter_width_));
  RETURN_IF_NOT_OK(ValidateScalar("Resample", "rolloff", rolloff_, {0, 1.0}, true, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> ResampleOperation::Build() {
  std::shared_ptr<ResampleOp> tensor_op =
    std::make_shared<ResampleOp>(orig_freq_, new_freq_, resample_method_, lowpass_filter_width_, rolloff_, beta_);
  return tensor_op;
}

Status ResampleOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["orig_freq"] = orig_freq_;
  args["new_freq"] = new_freq_;
  args["resample_method"] = resample_method_;
  args["lowpass_filter_width"] = lowpass_filter_width_;
  args["rolloff"] = rolloff_;
  args["beta"] = beta_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
