
#include "OURSdata/dataset/kernels/ir/vision/hwc_to_chw_ir.h"

#include "OURSdata/dataset/kernels/image/hwc_to_chw_op.h"

namespace ours {
namespace dataset {
namespace vision {
// HwcToChwOperation
HwcToChwOperation::HwcToChwOperation() : TensorOperation() {}

HwcToChwOperation::~HwcToChwOperation() = default;

std::string HwcToChwOperation::Name() const { return kHwcToChwOperation; }

Status HwcToChwOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> HwcToChwOperation::Build() { return std::make_shared<HwcToChwOp>(); }

Status HwcToChwOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<vision::HwcToChwOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
