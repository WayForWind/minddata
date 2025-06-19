

#include "OURSdata/dataset/audio/ir/kernels/complex_norm_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/complex_norm_op.h"

namespace ours {
namespace dataset {
namespace audio {
ComplexNormOperation::ComplexNormOperation(float power) : power_(power) {}

ComplexNormOperation::~ComplexNormOperation() = default;

Status ComplexNormOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("ComplexNorm", "power", power_));
  return Status::OK();
}

Status ComplexNormOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["power"] = power_;
  *out_json = args;
  return Status::OK();
}

std::shared_ptr<TensorOp> ComplexNormOperation::Build() {
  std::shared_ptr<ComplexNormOp> tensor_op = std::make_shared<ComplexNormOp>(power_);
  return tensor_op;
}

std::string ComplexNormOperation::Name() const { return kComplexNormOperation; }
}  // namespace audio
}  // namespace dataset
}  // namespace ours
