

#include "OURSdata/dataset/audio/ir/kernels/angle_ir.h"

#include "OURSdata/dataset/audio/kernels/angle_op.h"

namespace ours {
namespace dataset {
namespace audio {
// AngleOperation
AngleOperation::AngleOperation() = default;

Status AngleOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> AngleOperation::Build() {
  std::shared_ptr<AngleOp> tensor_op = std::make_shared<AngleOp>();
  return tensor_op;
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
