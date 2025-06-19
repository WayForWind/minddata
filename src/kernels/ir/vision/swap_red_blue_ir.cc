
#include "OURSdata/dataset/kernels/ir/vision/swap_red_blue_ir.h"

#if defined(ENABLE_OURSdata_PYTHON)
#include "OURSdata/dataset/kernels/image/swap_red_blue_op.h"
#endif

namespace ours {
namespace dataset {
namespace vision {
#if defined(ENABLE_OURSdata_PYTHON)
// SwapRedBlueOperation.
SwapRedBlueOperation::SwapRedBlueOperation() = default;

SwapRedBlueOperation::~SwapRedBlueOperation() = default;

std::string SwapRedBlueOperation::Name() const { return kSwapRedBlueOperation; }

Status SwapRedBlueOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> SwapRedBlueOperation::Build() {
  std::shared_ptr<SwapRedBlueOp> tensor_op = std::make_shared<SwapRedBlueOp>();
  return tensor_op;
}

Status SwapRedBlueOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<vision::SwapRedBlueOperation>();
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace ours
