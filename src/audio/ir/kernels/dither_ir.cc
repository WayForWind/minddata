

#include "OURSdata/dataset/audio/ir/kernels/dither_ir.h"

#include "OURSdata/dataset/audio/kernels/dither_op.h"

namespace ours {
namespace dataset {
namespace audio {
// DitherOperation
DitherOperation::DitherOperation(DensityFunction density_function, bool noise_shaping)
    : density_function_(density_function), noise_shaping_(noise_shaping) {
  random_op_ = true;
}

Status DitherOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DitherOperation::Build() {
  std::shared_ptr<DitherOp> tensor_op = std::make_shared<DitherOp>(density_function_, noise_shaping_);
  return tensor_op;
}

Status DitherOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["density_function"] = density_function_;
  args["noise_shaping"] = noise_shaping_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
