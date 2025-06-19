/
#include "OURSdata/dataset/kernels/ir/vision/decode_video_ir.h"

#include "OURSdata/dataset/kernels/image/decode_video_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// DecodeVideoOperation
DecodeVideoOperation::DecodeVideoOperation() {}

DecodeVideoOperation::~DecodeVideoOperation() = default;

std::string DecodeVideoOperation::Name() const { return kDecodeVideoOperation; }

Status DecodeVideoOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DecodeVideoOperation::Build() { return std::make_shared<DecodeVideoOp>(); }

Status DecodeVideoOperation::to_json(nlohmann::json *out_json) { return Status::OK(); }

Status DecodeVideoOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::DecodeVideoOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
