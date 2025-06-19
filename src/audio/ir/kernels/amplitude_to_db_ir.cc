

#include "OURSdata/dataset/audio/ir/kernels/amplitude_to_db_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/amplitude_to_db_op.h"

namespace ours {
namespace dataset {
namespace audio {
// AmplitudeToDBOperation
AmplitudeToDBOperation::AmplitudeToDBOperation(ScaleType stype, float ref_value, float amin, float top_db)
    : stype_(stype), ref_value_(ref_value), amin_(amin), top_db_(top_db) {}

AmplitudeToDBOperation::~AmplitudeToDBOperation() = default;

std::string AmplitudeToDBOperation::Name() const { return kAmplitudeToDBOperation; }

Status AmplitudeToDBOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AmplitudeToDB", "top_db", top_db_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("AmplitudeToDB", "amin", amin_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("AmplitudeToDB", "ref_value", ref_value_));

  return Status::OK();
}

std::shared_ptr<TensorOp> AmplitudeToDBOperation::Build() {
  std::shared_ptr<AmplitudeToDBOp> tensor_op = std::make_shared<AmplitudeToDBOp>(stype_, ref_value_, amin_, top_db_);
  return tensor_op;
}

Status AmplitudeToDBOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["stype"] = stype_;
  args["ref_value"] = ref_value_;
  args["amin"] = amin_;
  args["top_db"] = top_db_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
